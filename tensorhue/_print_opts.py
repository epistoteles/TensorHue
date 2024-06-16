from dataclasses import dataclass
from tensorhue.colors import ColorScheme


@dataclass
class __PrinterOptions:
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 200
    colorscheme: ColorScheme = ColorScheme()


PRINT_OPTS = __PrinterOptions()


# We could use **kwargs, but this will give better docs
def set_printoptions(
    threshold: int = None,
    edgeitems: int = None,
    linewidth: int = None,
    colorscheme: ColorScheme = None,
    accessible: bool = False,
):
    """Set options for printing. Items shamelessly taken from NumPy

    Args:
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 200). Thresholded matrices will
            ignore this parameter.
        colorscheme: The color scheme to use.
        accessible: Whether to use accessible mode or not (default = False).
    """
    if threshold is not None:
        assert isinstance(threshold, int)
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        assert isinstance(edgeitems, int)
        assert (
            edgeitems <= PRINT_OPTS.threshold // 2
        ), "edgeitems should not be larger than half the summarization threshold"
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        assert isinstance(linewidth, int)
        PRINT_OPTS.linewidth = linewidth
    if colorscheme is not None:
        assert isinstance(colorscheme, ColorScheme)
        PRINT_OPTS.colorscheme = colorscheme
    if accessible:
        raise NotImplementedError(
            "Accessible mode is not implemented yet. If you have a vision impairment and can provide feedback or beta-test color schemes please let us know."
        )
