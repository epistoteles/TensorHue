<div align="center">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/.github/tensorhue.png" alt="TensorHue" width="1152">
</div>
<br>
<div align="center">
  <img src="https://img.shields.io/badge/python-≥3.9-blue.svg">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/coverage-badge.svg">
  <img src="https://img.shields.io/pypi/dm/tensorhue">
  <img src="https://img.shields.io/badge/contributions-welcome-orange.svg">
</div>

> [!WARNING]
> TensorHue is currently in pre-alpha. We appreciate any feedback!

# TensorHue - tensors, visualized

TensorHue is a Python library that allows you to visualize tensors right in your console, making understanding and debugging tensor contents easier.

You can use it with your favorite tensor processing libraries, such as PyTorch, JAX*, and TensorFlow*.
_*coming soon_

TensorHue automagically detects which kind of tensor you are visualizing and adjusts accordingly:

<div align="center">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/.github/tensor_types.png" alt="tensor types" width="1152">
</div>

## Getting started

Using TensorHue is easy, simply import TensorHue after importing the library of your choice:

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

Numpy arrays can only be visualized with `tensorhue.viz(...)` (because np.array is immutable):

```python
np.array([1,2,3]).viz() ❌
tensorhue.viz(np.array([1,2,3])) ✅
```
## Custom colors

You can pass along your own ColorScheme when visualizing a specific tensor:

```python
from tensorhue import ColorScheme
from matplotlib import colormaps

cs = ColorScheme(colormap=colormaps['inferno'],
                 true_color=(10,10,10),
                 false_color=(20,20,20))
t.viz(cs)
```

Alternatively, you can overwrite the default ColorScheme:


```python
tensorhue.set_printoptions(colorscheme=cs)
```

## Advanced colors

By default, TensorHue normalizes numerical values between 0 and 1 and then applies the matplotlib colormap. If you want to use diverging colormaps such as `coolwarm` or `bwr` and the value 0 to be mapped to the middle of the colormap, you need to specify the normailzer, e.g. `matplotlib.colors.CenteredNorm`:

from matplotlib.colors import CenteredNorm

```python
cs = ColorScheme(colormap=colormaps['bwr'],
                 normalize=CenteredNorm(vcenter=0))
t.viz(cs)
```
