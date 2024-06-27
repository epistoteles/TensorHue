from dataclasses import dataclass
from tensorhue.colors import ColorScheme


@dataclass
class __PrinterOptions:
    colorscheme: ColorScheme = ColorScheme()
    edgeitems: int = 3


PRINT_OPTS = __PrinterOptions()


# We could use **kwargs, but this will give better docs
def set_printoptions(
    edgeitems: int = None,
    colorscheme: ColorScheme = None,
):
    """Set options for printing. Items shamelessly taken from NumPy

    Args:
        colorscheme: The color scheme to use.
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
    """
    if edgeitems is not None:
        assert isinstance(edgeitems, int)
        PRINT_OPTS.edgeitems = edgeitems
    if colorscheme is not None:
        assert isinstance(colorscheme, ColorScheme)
        PRINT_OPTS.colorscheme = colorscheme
