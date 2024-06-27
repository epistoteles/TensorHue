import os
import pytest
import torch
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorhue.viz import viz
from tensorhue.connectors._torch import _tensorhue_to_numpy_torch
from tensorhue.connectors._jax import _tensorhue_to_numpy_jax
from tensorhue.connectors._tensorflow import _tensorhue_to_numpy_tensorflow


@pytest.mark.parametrize(
    "tensor",
    [
        np.ones(10),
        _tensorhue_to_numpy_torch(torch.ones(10)),
        _tensorhue_to_numpy_jax(jnp.ones(10)),
        _tensorhue_to_numpy_tensorflow(tf.ones(10)),
    ],
)
def test_1d_tensor(tensor, capsys):
    viz(tensor)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 2
    assert out.count("▀") == 10
    assert out.split("\n")[-1] == f"shape = {tensor.shape}"


@pytest.mark.parametrize(
    "tensor",
    [
        np.ones((10, 10)),
        _tensorhue_to_numpy_torch(torch.ones(10, 10)),
        _tensorhue_to_numpy_jax(jnp.ones((10, 10))),
        _tensorhue_to_numpy_tensorflow(tf.ones((10, 10))),
    ],
)
def test_2d_tensor(tensor, capsys):
    viz(tensor)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 6
    assert out.count("▀") == 100 / 2
    assert out.split("\n")[-1] == f"shape = {tensor.shape}"


@pytest.mark.parametrize(
    "tensor",
    [
        np.ones(200),
        _tensorhue_to_numpy_torch(torch.ones(200)),
        _tensorhue_to_numpy_jax(jnp.ones(200)),
        _tensorhue_to_numpy_tensorflow(tf.ones(200)),
    ],
)
def test_1d_tensor_too_wide(tensor, capsys):
    viz(tensor)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert out.count(" ··· ") == 1
    assert out.count("▀") == 95
    assert out.split("\n")[-1] == f"shape = {tensor.shape}"


@pytest.mark.parametrize(
    "tensor",
    [
        np.ones((10, 200)),
        _tensorhue_to_numpy_torch(torch.ones(10, 200)),
        _tensorhue_to_numpy_jax(jnp.ones((10, 200))),
        _tensorhue_to_numpy_tensorflow(tf.ones((10, 200))),
    ],
)
def test_2d_tensor_too_wide(tensor, capsys):
    viz(tensor)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert out.count(" ··· ") == 5
    assert out.count("▀") == 950 / 2
    assert out.split("\n")[-1] == f"shape = {tensor.shape}"


@pytest.mark.parametrize(
    "tensor",
    [
        np.ones(10),
        _tensorhue_to_numpy_torch(torch.ones(10)),
        _tensorhue_to_numpy_jax(jnp.ones(10)),
        _tensorhue_to_numpy_tensorflow(tf.ones(10)),
    ],
)
def test_no_legend(tensor, capsys):
    viz(tensor, legend=False)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 1
    assert out.count("▀") == 10


@pytest.mark.parametrize("scale", [1, 2, 4, 8])
def test_scale(scale, capsys):
    tensor = np.ones((4, 4))
    viz(tensor, scale=scale)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert out.count("▀") == (8 * (scale**2))


@pytest.mark.parametrize("image_filename", os.listdir("./tests/test_resources/"))
@pytest.mark.parametrize("thumbnail", [True, False])
def test_viz_image(image_filename, thumbnail, capsys):
    filepath = "./tests/test_resources/" + image_filename
    image = Image.open(filepath)
    viz(image, thumbnail=thumbnail)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert out.count(" ··· ") == (0 if thumbnail else 300)
    assert len(out.split("\n")) == (100 if thumbnail else 600)
    assert out.count("▀") == (5000 if thumbnail else 28500)


@pytest.mark.parametrize("thumbnail", [True, False])
def test_viz_image_legend(thumbnail, capsys):
    filepath = "./tests/test_resources/test_image_rgba.png"
    image = Image.open(filepath)
    viz(image, legend=True, thumbnail=thumbnail)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert out.split("\n")[-1] == "size = (600, 600), mode = RGBA"
