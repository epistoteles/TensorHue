<div align="center">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/.github/tensorhue.png" alt="TensorHue" width="1152">
</div>
<br>
<div align="center">
  <img src="https://img.shields.io/badge/python-≥3.9-blue.svg">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/coverage-badge.svg">
  <img src="https://img.shields.io/pypi/dm/tensorhue">
  <a href="https://x.com/TensorHue" target="_blank"><img src="https://img.shields.io/twitter/follow/TensorHue"></a>
  <img src="https://img.shields.io/badge/contributions-welcome-orange.svg">
</div>

> [!WARNING]
> TensorHue is currently in alpha. Expect bugs. We appreciate any feedback!

# TensorHue - tensors, visualized

TensorHue is a Python library that allows you to visualize tensors right in your console, making understanding and debugging tensor contents easier.

You can use it with your favorite tensor processing libraries, such as PyTorch, JAX, and TensorFlow, and a large set of related libraries, including Numpy, Pillow, torchvision, and more.  

TensorHue automagically detects which kind of tensor you are visualizing and adjusts accordingly:

<div align="center">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/.github/tensor_types.png" alt="tensor types" width="1152">
</div>

## Getting started

Install TensorHue with pip:

```bash
pip install tensorhue
```

Using TensorHue is easy, simply import TensorHue *after* importing the library of your choice:

```python
import torch
import tensorhue
```

That's it! You can now visualize any tensor by calling .viz() on it in your Python console:

```python
t = torch.rand(20,20)
t.viz() ✅
```

You can also visualize them like this:

```python
tensorhue.viz(t) ✅
```

Numpy arrays can only be visualized with `tensorhue.viz(...)` (because np.ndarray is immutable):

```python
np.array([1,2,3]).viz() ❌
tensorhue.viz(np.array([1,2,3])) ✅
```

## Images

Pillow images can be visualized in RGB using `.viz()`:

```python
from torchvision.datasets import CIFAR10
dataset = CIFAR10('.', dowload=True)
img = dataset[0][0]
img.viz() ✅
```

<div align="center">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/.github/images.png" alt="image visualization" width="1000">
</div>

By default, images get downscaled to the size of your terminal, but you can make them even smaller if you want:

```python
img.viz(max_size=(40,40)) ✅
```

## Custom colors

You can pass along your own ColorScheme when visualizing a specific tensor:

```python
from tensorhue import ColorScheme
from matplotlib import colormaps

cs = ColorScheme(colormap=colormaps['inferno'],
                 true_color=(255,255,255),
                 false_color=(0,0,0))
t.viz(cs)
```

Alternatively, you can overwrite the default ColorScheme:


```python
tensorhue.set_printoptions(colorscheme=cs)
```

## Advanced colormaps and normalization

By default, TensorHue normalizes numerical values between 0 and 1 and then applies the matplotlib colormap. If you want to use diverging colormaps such as `coolwarm` or `bwr` and the value 0 to be mapped to the middle of the colormap, you need to specify the normailzer, e.g. `matplotlib.colors.CenteredNorm`:

```python
from matplotlib.colors import CenteredNorm
cs = ColorScheme(colormap=colormaps['bwr'],
                 normalize=CenteredNorm(vcenter=0))
t.viz(cs)
```

You can also specify the normalization range manually, for example when you want to visualize a confusion matrix where colors should be mapped to the range [0, 1], but the actual values of the tensor in the range [0.01, 0.73]:

```
conf_matrix.viz(vmin=0, vmax=1, scale=3)
```

<div align="center">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/.github/confusion_matrix.png" alt="confusion matrix" width="1000">
</div>

The `scale` parameter scales up the 'pixels' of the tensor so that small tensors are easier to view.
