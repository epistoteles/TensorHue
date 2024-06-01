<div align="center">
  <img src="https://raw.githubusercontent.com/epistoteles/tensorhue/main/.github/tensorhue.png" alt="TensorHue" width="1152">
</div>

<div align="center">
  <img src="https://img.shields.io/badge/python-≥v3.9-blue.svg">
  <img src="https://img.shields.io/badge/contributions-welcome-orange.svg">
</div>

<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-how-to-use">How To Use</a> •
  <a href="#-components">Components</a> •
  <a href="#-contributors">Contributors</a> •
  <a href="#%EF%B8%8F-license">License</a>
</p>

## TensorHue - tensors, visualized

TensorHue is a Python library that allows you to visualize tensors in your terminal, making understanding and debugging tensor contents easier.

You can use it with your favorite tensor processing libraries, such as PyTorch, JAX, and TensorFlow.

## Getting started

Using TensorHue is easy, simply import TensorHue after importing the relevant library:

```python
import torch
import tensorhue
```

That's it! You can now vizualize any tensor by calling .viz():

```python
t = torch.FloatTensor(20,20)
t.viz()
```
