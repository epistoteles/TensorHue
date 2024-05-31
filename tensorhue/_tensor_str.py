import dataclasses


@dataclasses.dataclass
class __PrinterOptions:
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 120
    colorscheme: str = "default"


PRINT_OPTS = __PrinterOptions()


# We could use **kwargs, but this will give better docs
def set_printoptions(
    threshold: int = None,
    edgeitems: int = None,
    linewidth: int = None,
    colorscheme: str = None,
):
    r"""Set options for printing. Items shamelessly taken from NumPy

    Args:
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        colorscheme: The color scheme to use. Intended to improve standard colors
            for people with color blindness. (any of `default`, `accessible`)

    Example::

        >>> TODO

    """

    if threshold is not None:
        assert isinstance(threshold, int)
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        assert isinstance(edgeitems, int)
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        assert isinstance(linewidth, int)
        PRINT_OPTS.linewidth = linewidth
    if colorscheme is not None:
        assert colorscheme in {"default", "accessible"}
        PRINT_OPTS.colorscheme = colorscheme


def get_printoptions() -> dict[str, any]:
    r"""Gets the current options for printing, as a dictionary that
    can be passed as ``**kwargs`` to set_printoptions().
    """
    return dataclasses.asdict(PRINT_OPTS)
