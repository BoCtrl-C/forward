from forward.utils import longest_path_algorithm, push_sources

import math

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Module, Parameter
from torch.nn.init import kaiming_uniform_, uniform_


class ForWard(Module):
    """PyTorch-compatible Module that allows the generation of feedforward
    neural networks from arbitrary DAGs. The input DAG is required to be a
    NetworkX DiGraph with integers as node IDs.
    """
    
    def __init__(
        self,
        dag,
        bias=True,
        activation=F.relu,
        initialization=lambda w: kaiming_uniform_(w, a=math.sqrt(5)),
        dropout=0.,
        stand_alone=False, # when deployed as a stand-alone module, this flag
                           # disables the last layer's nonlinearities
        seed=1
    ):
        super().__init__()

        # set seed for reproducibility
        torch.manual_seed(seed)

        # assign layers to nodes
        L = longest_path_algorithm(dag)
        
        # push source nodes into the first layer
        L = push_sources(dag, L)

        # retrieve source and sink nodes
        sources = L[0]
        sinks = L[-1]

        layer_preds = []
        weights = []
        masks = []
        for i, layer in enumerate(L[1:]):
            # store predecessors of nodes in the current layer
            preds = set()
            for v in layer:
                preds |= set(dag.predecessors(v))
            preds = sorted(list(preds)) # NOTE: integer node IDs are expected

            # allocate memory for the layer weights
            W = Parameter(torch.zeros(
                len(preds) + 1 if bias else len(preds),
                len(layer)
            ))

            # build a mask to restore the complex topology
            M = Parameter(torch.zeros_like(W), requires_grad=False)
            for j, v in enumerate(layer):
                for pred in dag.predecessors(v):
                    M[preds.index(pred),j] = 1.
            
            # initialize weights
            for j in range(M.shape[1]):
                fan_in = M[:,j].sum().int()

                w = torch.zeros(fan_in, 1)
                initialization(w.T)
                W.data[M[:,j].bool(),j] = w.squeeze()
            
                # initialize the neuron bias
                if bias:
                    M[-1,j] = 1.
                    bound = 1/math.sqrt(fan_in) if fan_in > 0 else 0
                    uniform_(W[-1,j], -bound, bound)
            
            self.register_parameter('M' + str(i + 1), M)
            self.register_parameter('W' + str(i + 1), W)

            layer_preds.append(preds)
            weights.append(W)
            masks.append(M)
        
        self.in_features = len(sources)
        self.out_features = len(sinks)
        self.dag = dag
        self.L = L
        self.sources = sources
        self.sinks = sinks
        self.bias = bias
        self.layer_preds = layer_preds
        self.weights = weights
        self.masks = masks
        self.activation = activation
        self.dropout = Dropout(p=dropout)
        self.stand_alone = stand_alone
    
    def forward(self, x):
        if x.shape[1] != self.in_features:
            raise RuntimeError(
                'the model expects {:d} input features'.format(self.in_features)
            )
               
        # initialize activations
        activations = torch.zeros(
            x.shape[0],
            self.dag.number_of_nodes(),
            device=x.device
        ) # NOTE: the space complexity is O(N)
        activations[:,self.sources] = x

        # forward
        for i, (layer, preds, W, M) in enumerate(zip(
            self.L[1:],
            self.layer_preds,
            self.weights,
            self.masks
        )):
            out = torch.mm(
                torch.cat([
                    self.dropout(activations[:,preds]),
                    torch.ones(x.shape[0], 1, device=x.device) if self.bias
                        else torch.ones(x.shape[0], 0, device=x.device)
                ], dim=1),
                M*W
            )

            if not self.stand_alone or i + 2 < len(self.L):
                out = self.activation(out)

            activations[:,layer] = out
        
        out = activations[:,self.sinks]

        return out

    def _forward(self, x):
        """Use only if dropout has to ignore source activations.
        """
        
        if x.shape[1] != self.in_features:
            raise RuntimeError(
                'the model expects {:d} input features'.format(self.in_features)
            )
               
        # initialize activations
        activations = torch.zeros(
            x.shape[0],
            self.dag.number_of_nodes(),
            device=x.device
        ) # NOTE: the space complexity is O(N)
        activations[:,self.sources] = x

        # forward
        for i, (layer, preds, W, M) in enumerate(zip(
            self.L[1:],
            self.layer_preds,
            self.weights,
            self.masks
        )):
            out = torch.mm(
                torch.cat([
                    activations[:,preds] if i == 0
                        else self.dropout(activations[:,preds]),
                    torch.ones(x.shape[0], 1, device=x.device) if self.bias
                        else torch.ones(x.shape[0], 0, device=x.device)
                ], dim=1),
                M*W
            )

            if not self.stand_alone or i + 2 < len(self.L):
                out = self.activation(out)

            activations[:,layer] = out
        
        out = activations[:,self.sinks]

        return out