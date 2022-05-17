# 4Ward - Make your Graph a Feedforward Neural Network

With 4Ward, directed acyclic graphs (DAGs) characterized by complex topologies can be easily transformed into feedforward neural networks deployable as PyTorch Modules.

## Get Started

Install 4Ward by typing:
```
pip install git+https://github.com/BoCtrl-C/forward.git
```

The main 4Ward class can be imported through:
```python
from forward.models import ForWard
```
Finally, deploy your new PyTorch Module wherever you want! For instance:
```python
from torch.nn import Sequential


dag = ...

ffnn = ForWard(dag)

model = Sequential(
    ...,
    ffnn,
    ...
)
```

## Examples

A runnable Jupyter Notebook that makes use of 4Ward to classify MNIST images can be found inside the `examples` directory.
