import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from rich.color_triplet import ColorTriplet
from tensorhue.colors import ColorScheme
from tensorhue.viz import viz


def pride():
    pride_colors = [
        ColorTriplet(228, 3, 3),
        ColorTriplet(255, 140, 0),
        ColorTriplet(255, 237, 0),
        ColorTriplet(0, 128, 38),
        ColorTriplet(0, 76, 255),
        ColorTriplet(115, 41, 130),
    ]
    pride_cm = LinearSegmentedColormap.from_list(colors=[c.normalized for c in pride_colors], name="pride")
    pride_cs = ColorScheme(colormap=pride_cm)
    arr = np.repeat(np.linspace(0, 1, 6).reshape(-1, 1), 10, axis=1)
    viz(arr, colorscheme=pride_cs)
