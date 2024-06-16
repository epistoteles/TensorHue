import sys
from rich.console import Console
import numpy as np
from tensorhue.colors import COLORS, ColorScheme
from tensorhue._print_opts import PRINT_OPTS, set_printoptions
from tensorhue.numpy import NumpyArrayWrapper
from tensorhue.torch import _tensorhue_to_numpy_torch
from tensorhue.eastereggs import pride
from tensorhue.viz import viz, _viz


__version__ = "0.0.7"  # single source of version truth

__all__ = ["set_printoptions", "viz", "pride"]

# automagically set up TensorHue
setattr(NumpyArrayWrapper, "viz", _viz)
if "torch" in sys.modules:
    torch = sys.modules["torch"]
    setattr(torch.Tensor, "viz", _viz)
    setattr(torch.Tensor, "_tensorhue_to_numpy", _tensorhue_to_numpy_torch)
