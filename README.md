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

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2209.02037,
  doi = {10.48550/ARXIV.2209.02037},
  url = {https://arxiv.org/abs/2209.02037},
  author = {Boccato, Tommaso and Ferrante, Matteo and Duggento, Andrea and Toschi, Nicola},
  keywords = {Neural and Evolutionary Computing (cs.NE), Disordered Systems and Neural Networks (cond-mat.dis-nn), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Physical sciences, FOS: Physical sciences},
  title = {4Ward: a Relayering Strategy for Efficient Training of Arbitrarily Complex Directed Acyclic Graphs},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```
