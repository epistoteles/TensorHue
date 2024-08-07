import sys
import inspect
from tensorhue.colors import COLORS, ColorScheme
from tensorhue._print_opts import PRINT_OPTS, set_printoptions
from tensorhue.connectors._numpy import NumpyArrayWrapper
from tensorhue.connectors._torch import _tensorhue_to_numpy_torch
from tensorhue.connectors._jax import _tensorhue_to_numpy_jax
from tensorhue.connectors._tensorflow import _tensorhue_to_numpy_tensorflow
from tensorhue.connectors._pillow import _tensorhue_to_numpy_pillow
from tensorhue.eastereggs import pride
from tensorhue.viz import viz, _viz, _viz_image


__version__ = "0.0.16"  # single source of version truth

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
    if "DeviceArrayBase" in {x[0] for x in inspect.getmembers(jaxlib.xla_extension)}:  # jax < 0.4.X
        setattr(jaxlib.xla_extension.DeviceArrayBase, "viz", _viz)
        setattr(jaxlib.xla_extension.DeviceArrayBase, "_tensorhue_to_numpy", _tensorhue_to_numpy_jax)
    if "ArrayImpl" in {
        x[0] for x in inspect.getmembers(jaxlib.xla_extension)
    }:  # jax >= 0.4.X (not sure about the exact version this changed)
        setattr(jaxlib.xla_extension.ArrayImpl, "viz", _viz)
        setattr(jaxlib.xla_extension.ArrayImpl, "_tensorhue_to_numpy", _tensorhue_to_numpy_jax)
if "tensorflow" in sys.modules:
    tensorflow = sys.modules["tensorflow"]
    setattr(tensorflow.Tensor, "viz", _viz)
    setattr(tensorflow.Tensor, "_tensorhue_to_numpy", _tensorhue_to_numpy_tensorflow)
    composite_tensor = sys.modules["tensorflow.python.framework.composite_tensor"]
    setattr(composite_tensor.CompositeTensor, "viz", _viz)
    setattr(composite_tensor.CompositeTensor, "_tensorhue_to_numpy", _tensorhue_to_numpy_tensorflow)
if "PIL" in sys.modules:
    PIL = sys.modules["PIL"]
    setattr(PIL.Image.Image, "viz", _viz_image)
    setattr(PIL.Image.Image, "_tensorhue_to_numpy", _tensorhue_to_numpy_pillow)
