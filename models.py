from utils import longest_path_algorithm, push_sources

from typing import Union, List, Dict, Any, Optional, cast

import math

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, Parameter, ReLU, Sequential
from torch.nn.init import kaiming_uniform_


class ForWard(Module):
    """TODO: add documentation
    """
    
    def __init__(
        self,
        dag,
        bias=True,
        dropout=0.,
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
            if bias: M[-1] = 1.
            self.register_parameter('M' + str(i + 1), M)
            
            # initialize weights
            for j in range(M.shape[1]):
                w = torch.zeros(M[:,j].sum().int(), 1)
                kaiming_uniform_(w, a=math.sqrt(5))
                W.data[M[:,j].bool(),j] = w.squeeze()
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
        self.dropout = Dropout(p=dropout)
    
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
        ) # NOTE: space complexity number of nodes FIXME:
        activations[:,self.sources] = x

        # forward
        for layer, preds, W, M in zip(
            self.L[1:],
            self.layer_preds,
            self.weights,
            self.masks
        ):
            activations[:,layer] = F.relu(torch.mm(
                torch.cat([
                    self.dropout(activations[:,preds]),
                    torch.ones(x.shape[0], 1, device=x.device) if self.bias
                        else torch.ones(x.shape[0], 0, device=x.device)
                ], dim=1),
                M*W
            ))
        
        out = activations[:,self.sinks]

        return out