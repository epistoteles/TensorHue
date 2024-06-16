import pytest
from rich.color_triplet import ColorTriplet
import numpy as np

from tensorhue.colors import ColorScheme, COLORS
from matplotlib.colors import Colormap
from matplotlib import colormaps


def test_COLORS():
    for key, value in COLORS.items():
        assert isinstance(key, str)
        assert isinstance(value, ColorTriplet)


def test_ColorScheme():
    cs = ColorScheme()

    assert isinstance(cs.colormap, Colormap)
    assert cs.masked_color == COLORS["masked"]
    assert cs.true_color == COLORS["true"]
    assert np.array_equal(cs.colormap.get_bad()[:3], np.array(cs.masked_color.normalized))

    values1 = np.array([-0.5, 0.0, 0.5, 0.75])
    result1 = cs(values1)
    values2 = values1 * 2
    result2 = cs(values2)
    assert np.array_equal(result1, result2)

    cs.colormap = colormaps["cividis"]
    assert np.array_equal(cs.colormap.get_bad()[:3], np.array(cs.masked_color.normalized))

    cs.masked_color = COLORS["black"]
    assert np.array_equal(cs.colormap.get_bad()[:3], np.array([0, 0, 0]))

    cs.inf_color = COLORS["black"]
    assert np.array_equal(cs.colormap.get_over()[:3], np.array([0, 0, 0]))

    cs.ninf_color = COLORS["black"]
    assert np.array_equal(cs.colormap.get_under()[:3], np.array([0, 0, 0]))

    bool_array = np.array([True, False])
    assert np.array_equal(cs(bool_array), np.array([cs.true_color, cs.false_color]))
