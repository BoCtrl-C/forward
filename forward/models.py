from forward.utils import longest_path_algorithm, push_sources

import math

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Module, Parameter
#from torch.nn.init import kaiming_uniform_


import warnings
from torch.nn.init import _calculate_correct_fan, calculate_gain
def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    mode='fan_out'#TODO: remove
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            kaiming_uniform_,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity)

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    #print('tensor:', tensor.shape)
    #print('fan:', fan)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class ForWard(Module):
    """TODO: add documentation
    """
    
    def __init__(
        self,
        dag,
        bias=True,
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
        ) # NOTE: space complexity number of nodes FIXME:
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
                out = F.relu(out)

            activations[:,layer] = out
        
        out = activations[:,self.sinks]

        return out