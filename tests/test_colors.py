import pytest
from rich.color_triplet import ColorTriplet
import numpy as np

from tensorhue.colors import ColorGradient, ColorScheme, COLORS


def test_COLORS():
    for key, value in COLORS.items():
        assert isinstance(key, str)
        assert isinstance(value, ColorTriplet)


# def test_ColorScheme():
#     cs = ColorScheme(
#         gradient=ColorGradient(
#             [(0.0, ColorTriplet(0, 0, 0)), (0.5, ColorTriplet(255, 255, 255)), (1.0, ColorTriplet(0, 0, 0))]
#         )
#     )
#     assert cs.gradient.gradient == [
#         (0.0, ColorTriplet(0, 0, 0)),
#         (0.5, ColorTriplet(255, 255, 255)),
#         (1.0, ColorTriplet(0, 0, 0)),
#     ]
#     assert cs.masked_color == COLORS["masked"]
#     assert cs.true_color == COLORS["true"]

#     values = np.array([0.0, 0.5, 0.75])
#     result = cs.calculate_gradient_color_vectorized(values)
#     assert np.array_equal(result, np.array([[0, 255, 127], [0, 255, 127], [0, 255, 127]]))
