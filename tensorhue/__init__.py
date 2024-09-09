import sys
import inspect
from tensorhue.colors import COLORS, ColorScheme
from tensorhue._print_opts import PRINT_OPTS, set_printoptions
from tensorhue.eastereggs import pride
from tensorhue.viz import viz, _viz, _viz_image


__version__ = "0.1.0"  # single source of version truth

__all__ = ["set_printoptions", "viz", "pride"]


# show deprecation warning for t.viz() usage
# delete everything below this line after version 0.2.0


def _viz_is_deprecated(self):
    raise DeprecationWarning("The tensor.viz() function has been deprecated. Please use tensorhue.viz(tensor) instead.")


if "torch" in sys.modules:
    torch = sys.modules["torch"]
    setattr(torch.Tensor, "viz", _viz_is_deprecated)
if "jax" in sys.modules:
    jax = sys.modules["jax"]
    setattr(jax.Array, "viz", _viz_is_deprecated)
    jaxlib = sys.modules["jaxlib"]
    if "DeviceArrayBase" in {x[0] for x in inspect.getmembers(jaxlib.xla_extension)}:  # jax < 0.4.X
        setattr(jaxlib.xla_extension.DeviceArrayBase, "viz", _viz_is_deprecated)
    if "ArrayImpl" in {
        x[0] for x in inspect.getmembers(jaxlib.xla_extension)
    }:  # jax >= 0.4.X (not sure about the exact version this changed)
        setattr(jaxlib.xla_extension.ArrayImpl, "viz", _viz_is_deprecated)
if "tensorflow" in sys.modules:
    tensorflow = sys.modules["tensorflow"]
    setattr(tensorflow.Tensor, "viz", _viz_is_deprecated)
    composite_tensor = sys.modules["tensorflow.python.framework.composite_tensor"]
    setattr(composite_tensor.CompositeTensor, "viz", _viz_is_deprecated)
if "PIL" in sys.modules:
    PIL = sys.modules["PIL"]
    setattr(PIL.Image.Image, "viz", _viz_is_deprecated)
