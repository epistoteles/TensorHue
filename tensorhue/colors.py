from dataclasses import dataclass, field
from rich.color_triplet import ColorTriplet
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Union, Any, Iterator
from scipy.interpolate import interp1d


COLORS = {
    "masked": ColorTriplet(140, 140, 140),  # medium grey
    "default_dark": ColorTriplet(64, 17, 159),  # dark purple
    "default_medium": ColorTriplet(255, 55, 140),  # pink
    "default_bright": ColorTriplet(255, 210, 240),  # light rose
    "true": ColorTriplet(255, 80, 80),  # green
    "false": ColorTriplet(125, 215, 82),  # red
    "accessible_true": ColorTriplet(255, 80, 80),  # TODO
    "accessible_false": ColorTriplet(125, 215, 82),  # TODO
    "black": ColorTriplet(0, 0, 0),  # black
    "white": ColorTriplet(255, 255, 255),  # white
}


@dataclass
class ColorGradient:
    gradient: List[Tuple[float, ColorTriplet]] = field(
        default_factory=lambda: [
            (0.0, COLORS["default_dark"]),
            (0.7, COLORS["default_medium"]),
            (1.0, COLORS["default_bright"]),
        ]
    )

    def __post_init__(self):
        if len(self.gradient) < 2:
            raise ValueError("ColorGradient must have at least 2 points")
        self.gradient = [(float(pos), color) for pos, color in self.gradient]
        pos_values = [pos for pos, _ in self.gradient]
        if len(set(pos_values)) != len(pos_values):
            raise ValueError("ColorGradient must have unique position values")
        if 0.0 not in pos_values:
            raise ValueError("ColorGradient must include a color for position 0.0")
        if 1.0 not in pos_values:
            raise ValueError("ColorGradient must include a color for position 1.0")
        if max(pos_values) > 1.0 or min(pos_values) < 0.0:
            raise ValueError("ColorGradient positions must be between 0.0 and 1.0")
        self.gradient = sorted(self.gradient, key=lambda x: x[0])

    def __iter__(self) -> Iterator[Tuple[float, ColorTriplet]]:
        return iter(self.gradient)


@dataclass
class ColorScheme:
    gradient: ColorGradient = field(default_factory=ColorGradient)
    masked_color: ColorTriplet = field(default_factory=lambda: COLORS["masked"])
    true_color: ColorTriplet = field(default_factory=lambda: COLORS["true"])
    false_color: ColorTriplet = field(default_factory=lambda: COLORS["false"])
    inf_color: ColorTriplet = field(default_factory=lambda: COLORS["white"])
    ninf_color: ColorTriplet = field(default_factory=lambda: COLORS["black"])

    def calculate_gradient_color_vectorized(
        self, value_array: Union[NDArray[np.number], NDArray[bool]]
    ) -> Union[NDArray[np.uint8], Any]:
        """
        Calculate the gradient color for each value in the input array using vectorized interpolation.

        Args:
            value_array (NDArray[np.float64]): The input array of values.

        Returns:
            Union[Any, NDArray[np.uint8]]: The calculated gradient color for each value in the input array.
        """
        positions = [pos for pos, _ in self.gradient]
        color_data = np.array([[color.red, color.green, color.blue] for _, color in self.gradient])
        interp_functions = [interp1d(positions, channel) for channel in color_data.T]
        flat_values = value_array.flatten()
        interpolated_colors = np.stack([func(flat_values) for func in interp_functions], axis=0)
        return interpolated_colors.reshape((3,) + value_array.shape).astype(np.uint8)
