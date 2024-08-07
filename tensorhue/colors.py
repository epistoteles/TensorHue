from __future__ import annotations
import warnings

from rich.color_triplet import ColorTriplet
from matplotlib import colormaps
from matplotlib.colors import Colormap, Normalize, CenteredNorm
import numpy as np


COLORS = {
    "masked": ColorTriplet(127, 127, 127),  # medium grey
    "default_dark": ColorTriplet(64, 17, 159),  # dark purple
    "default_medium": ColorTriplet(255, 55, 140),  # pink
    "default_bright": ColorTriplet(255, 210, 240),  # light rose
    "true": ColorTriplet(125, 215, 82),  # green
    "false": ColorTriplet(255, 80, 80),  # red
    "accessible_true": ColorTriplet(255, 255, 255),  # TODO
    "accessible_false": ColorTriplet(0, 0, 0),  # TODO
    "black": ColorTriplet(0, 0, 0),  # black
    "white": ColorTriplet(255, 255, 255),  # white
}


class ColorScheme:
    def __init__(
        self,
        colormap: Colormap = colormaps["magma"],
        normalize: Normalize = Normalize(),
        masked_color: ColorTriplet = COLORS["masked"],
        true_color: ColorTriplet = COLORS["true"],
        false_color: ColorTriplet = COLORS["false"],
        inf_color: ColorTriplet = COLORS["white"],
        ninf_color: ColorTriplet = COLORS["black"],
    ):
        self._colormap = colormap
        self.normalize = normalize
        self._masked_color = masked_color
        self.true_color = true_color
        self.false_color = false_color
        self._inf_color = inf_color
        self._ninf_color = ninf_color

        self.colormap.set_extremes(
            bad=self.masked_color.normalized, under=self.ninf_color.normalized, over=self.inf_color.normalized
        )

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        self._colormap = value
        self._colormap.set_extremes(
            bad=self._masked_color.normalized, under=self._ninf_color.normalized, over=self._inf_color.normalized
        )

    @property
    def masked_color(self):
        return self._masked_color

    @masked_color.setter
    def masked_color(self, value):
        self._masked_color = value
        self._colormap.set_bad(value.normalized)

    @property
    def inf_color(self):
        return self._inf_color

    @inf_color.setter
    def inf_color(self, value):
        self._inf_color = value
        self._colormap.set_over(value.normalized)

    @property
    def ninf_color(self):
        return self._ninf_color

    @ninf_color.setter
    def ninf_color(self, value):
        self._ninf_color = value
        self._colormap.set_under(value.normalized)

    def __call__(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if data.dtype == "bool":
            true_values = np.array(self.true_color, dtype=np.uint8)
            false_values = np.array(self.false_color, dtype=np.uint8)
            return np.where(data[..., np.newaxis], true_values, false_values)
        data_noinf = np.where(np.isinf(data), np.nan, data)
        if "vmin" not in kwargs:
            vmin = np.nanmin(data_noinf)
        else:
            vmin = float(kwargs["vmin"])
        if "vmax" not in kwargs:
            vmax = np.nanmax(data_noinf)
        else:
            vmax = float(kwargs["vmax"])
        if isinstance(self.normalize, CenteredNorm):
            vcenter = self.normalize.vcenter
            diff_vmin = vmin - vcenter
            diff_vmax = vmax - vcenter
            max_abs_diff = max(abs(diff_vmin), abs(diff_vmax))
            vmin = vcenter - max_abs_diff
            vmax = vcenter + max_abs_diff
            if "vmin" in kwargs and "vmax" in kwargs:
                warnings.warn(
                    f"You shouldn't specify both 'vmin' and 'vmax' when using CenteredNorm. 'vmin' and 'vmax' must be symmetric around 'vcenter' and are thus inferred from a single value. Using: {vmin=}, {vcenter=}, {vmax=}."
                )
        self.normalize.vmin = vmin
        self.normalize.vmax = vmax
        return self.colormap(self.normalize(data), bytes=True)

    def __repr__(self):
        return (
            f"ColorScheme(\n"
            f"    colormap={self._colormap},\n"
            f"    normalize={self.normalize},\n"
            f"    masked_color={self._masked_color},\n"
            f"    true_color={self.true_color},\n"
            f"    false_color={self.false_color},\n"
            f"    inf_color={self._inf_color},\n"
            f"    ninf_color={self._ninf_color}\n"
            f")"
        )
