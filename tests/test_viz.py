import torch
import numpy as np
from tensorhue.viz import viz
from tensorhue._torch import _tensorhue_to_numpy_torch


def test_1d_tensor_numpy(capsys):
    n = np.ones(10)
    viz(n)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 2
    assert out.count("▀") == 10
    assert out.split("\n")[-1] == f"shape = {n.shape}"


def test_2d_tensor_numpy(capsys):
    n = np.ones((10, 10))
    viz(n)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 6
    assert out.count("▀") == 50
    assert out.split("\n")[-1] == f"shape = {n.shape}"


def test_1d_tensor_torch(capsys):
    t = torch.ones(10)
    n = _tensorhue_to_numpy_torch(t)
    viz(n)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 2
    assert out.count("▀") == 10
    assert out.split("\n")[-1] == f"shape = {n.shape}"


def test_2d_tensor_torch(capsys):
    t = torch.ones(10, 10)
    n = _tensorhue_to_numpy_torch(t)
    viz(n)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 6
    assert out.count("▀") == 50
    assert out.split("\n")[-1] == f"shape = {n.shape}"


def test_no_legend(capsys):
    n = np.ones(10)
    viz(n, legend=False)
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 1
    assert out.count("▀") == 10
