from __future__ import annotations

from dataclasses import dataclass, field
from rich.color_triplet import ColorTriplet
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap, Normalize


COLORS = {
    "masked": ColorTriplet(140, 140, 140),  # medium grey
    "default_dark": ColorTriplet(64, 17, 159),  # dark purple
    "default_medium": ColorTriplet(255, 55, 140),  # pink
    "default_bright": ColorTriplet(255, 210, 240),  # light rose
    "true": ColorTriplet(125, 215, 82),  # green
    "false": ColorTriplet(255, 80, 80),  # red
    "accessible_true": ColorTriplet(255, 80, 80),  # TODO
    "accessible_false": ColorTriplet(125, 215, 82),  # TODO
    "black": ColorTriplet(0, 0, 0),  # black
    "white": ColorTriplet(255, 255, 255),  # white
}


@dataclass
class ColorScheme:
    _colormap: Colormap = field(default_factory=lambda: colormaps["magma"])
    normalize: Normalize = field(default_factory=Normalize)
    _masked_color: ColorTriplet = field(default_factory=lambda: COLORS["masked"])
    true_color: ColorTriplet = field(default_factory=lambda: COLORS["true"])
    false_color: ColorTriplet = field(default_factory=lambda: COLORS["false"])
    _inf_color: ColorTriplet = field(default_factory=lambda: COLORS["white"])
    _ninf_color: ColorTriplet = field(default_factory=lambda: COLORS["black"])

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        self._colormap = value
        self._colormap.set_extreme(bad=self._masked_color, under=self._ninf_color, over=self._inf_color)

    @property
    def masked_color(self):
        return self._masked_color

    @masked_color.setter
    def masked_color(self, value):
        self._masked_color = value
        self._colormap.set_bad(value)

    @property
    def inf_color(self):
        return self._inf_color

    @inf_color.setter
    def inf_color(self, value):
        self._inf_color = value
        self._colormap.set_over(value)

    @property
    def ninf_color(self):
        return self._ninf_color

    @ninf_color.setter
    def ninf_color(self, value):
        self._ninf_color = value
        self._colormap.set_under(value)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.colormap(self.normalize(data), bytes=True)

    def __repr__(self):
        return (
            f"ColorScheme(\n"
            f"    colormap={self.colormap},\n"
            f"    normalize={self.normalize},\n"
            f"    masked_color={self._masked_color},\n"
            f"    true_color={self.true_color},\n"
            f"    false_color={self.false_color},\n"
            f"    inf_color={self._inf_color},\n"
            f"    ninf_color={self._ninf_color}\n"
            f")"
        )
