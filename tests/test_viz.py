import pytest
import torch
import numpy as np
from tensorhue.viz import viz
from tensorhue.connectors._torch import _tensorhue_to_numpy_torch


@pytest.mark.parametrize("tensor", [np.ones(10), _tensorhue_to_numpy_torch(torch.ones(10))])
def test_1d_tensor(tensor, capsys):
    viz(tensor)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 2
    assert out.count("▀") == 10
    assert out.split("\n")[-1] == f"shape = {tensor.shape}"


@pytest.mark.parametrize("tensor", [np.ones((10, 10)), _tensorhue_to_numpy_torch(torch.ones(10, 10))])
def test_2d_tensor(tensor, capsys):
    viz(tensor)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 6
    assert out.count("▀") == 100 / 2
    assert out.split("\n")[-1] == f"shape = {tensor.shape}"


@pytest.mark.parametrize("tensor", [np.ones(200), _tensorhue_to_numpy_torch(torch.ones(200))])
def test_1d_tensor_too_wide(tensor, capsys):
    viz(tensor)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert out.count(" ··· ") == 1
    assert out.count("▀") == 95
    assert out.split("\n")[-1] == f"shape = {tensor.shape}"


@pytest.mark.parametrize("tensor", [np.ones((10, 200)), _tensorhue_to_numpy_torch(torch.ones(10, 200))])
def test_2d_tensor_too_wide(tensor, capsys):
    viz(tensor)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert out.count(" ··· ") == 5
    assert out.count("▀") == 950 / 2
    assert out.split("\n")[-1] == f"shape = {tensor.shape}"


@pytest.mark.parametrize("tensor", [np.ones(10), _tensorhue_to_numpy_torch(torch.ones(10))])
def test_no_legend(tensor, capsys):
    viz(tensor, legend=False)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 1
    assert out.count("▀") == 10
