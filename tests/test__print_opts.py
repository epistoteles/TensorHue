from tensorhue._print_opts import set_printoptions, PRINT_OPTS
from tensorhue.colors import ColorScheme


def test_default_print_opts():
    assert PRINT_OPTS.edgeitems == 3
    assert isinstance(PRINT_OPTS.colorscheme, ColorScheme)


def test_set_printopts():
    set_printoptions(edgeitems=42)
    assert PRINT_OPTS.edgeitems == 42
    cs = ColorScheme(true_color=(0, 0, 0))
    set_printoptions(colorscheme=cs)
    assert PRINT_OPTS.colorscheme == cs
    assert PRINT_OPTS.edgeitems == 42
