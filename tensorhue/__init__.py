import sys
from rich.console import Console
import tensorhue._numpy as np
from tensorhue.colors import COLORS, ColorScheme
from tensorhue._print_opts import PRINT_OPTS, set_printoptions
from tensorhue._numpy import NumpyArrayWrapper
from tensorhue._torch import _tensorhue_to_numpy_torch
from tensorhue._jax import _tensorhue_to_numpy_jax
from tensorhue.eastereggs import pride
from tensorhue.viz import viz, _viz


__version__ = "0.0.9"  # single source of version truth

__all__ = ["set_printoptions", "viz", "pride"]

# automagically set up TensorHue
setattr(NumpyArrayWrapper, "viz", _viz)
if "torch" in sys.modules:
    torch = sys.modules["torch"]
    setattr(torch.Tensor, "viz", _viz)
    setattr(torch.Tensor, "_tensorhue_to_numpy", _tensorhue_to_numpy_torch)
if "jax" in sys.modules:
    jax = sys.modules["jax"]
    setattr(jax.Array, "viz", _viz)
    setattr(jax.Array, "_tensorhue_to_numpy", _tensorhue_to_numpy_jax)
    jaxlib = sys.modules["jaxlib"]
    setattr(jaxlib.xla_extension.DeviceArrayBase, "viz", _viz)
    setattr(jaxlib.xla_extension.DeviceArrayBase, "_tensorhue_to_numpy", _tensorhue_to_numpy_jax)
