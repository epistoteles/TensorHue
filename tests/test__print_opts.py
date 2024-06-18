from tensorhue._print_opts import set_printoptions, PRINT_OPTS
from tensorhue.colors import ColorScheme


def test_default_print_opts():
    assert PRINT_OPTS.threshold > 100
    assert isinstance(PRINT_OPTS.colorscheme, ColorScheme)


def test_set_printopts():
    set_printoptions(threshold=1234)
    assert PRINT_OPTS.threshold == 1234
    set_printoptions(edgeitems=4)
    assert PRINT_OPTS.edgeitems == 4
    assert PRINT_OPTS.threshold == 1234
