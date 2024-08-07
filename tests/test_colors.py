from rich.color_triplet import ColorTriplet
import numpy as np
from matplotlib.colors import Colormap, CenteredNorm
from matplotlib import colormaps
from tensorhue.colors import ColorScheme, COLORS


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


def test_vmin_vmax():
    cs = ColorScheme(colormap=colormaps["magma"])

    values1 = np.array([-0.5, 0.0, 0.5, 0.75])

    result1 = cs(values1)
    assert np.array_equal(
        result1,
        np.array([[0, 0, 3, 255], [140, 41, 128, 255], [253, 159, 108, 255], [251, 252, 191, 255]], dtype=np.uint8),
    )

    result2 = cs(values1, vmin=-0.5)
    assert np.array_equal(result1, result2)

    result3 = cs(values1, vmax=0.75)
    assert np.array_equal(result1, result3)

    result4 = cs(values1, vmin=-0.5, vmax=0.75)
    assert np.array_equal(result1, result4)

    result5 = cs(values1, vmin=-1)
    assert np.array_equal(
        result5,
        np.array([[94, 23, 127, 255], [211, 66, 109, 255], [254, 187, 128, 255], [251, 252, 191, 255]], dtype=np.uint8),
    )

    result6 = cs(values1, vmax=0.4)
    assert np.array_equal(
        result6,
        np.array([[0, 0, 3, 255], [205, 63, 112, 255], [255, 255, 255, 255], [255, 255, 255, 255]], dtype=np.uint8),
    )

    cs = ColorScheme(colormap=colormaps["bwr"], normalize=CenteredNorm())

    result7 = cs(values1)
    assert np.array_equal(
        result7,
        np.array([[84, 84, 255, 255], [255, 254, 254, 255], [255, 84, 84, 255], [255, 0, 0, 255]], dtype=np.uint8),
    )

    result8 = cs(values1, vmin=-1)
    assert np.array_equal(
        result8,
        np.array(
            [[128, 128, 255, 255], [255, 254, 254, 255], [255, 126, 126, 255], [255, 62, 62, 255]], dtype=np.uint8
        ),
    )
