import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from rich.color_triplet import ColorTriplet
from tensorhue.colors import ColorScheme
from tensorhue.viz import viz, get_terminal_size


def pride(width: int = None):
    """
    Prints a pride flag in the terminal

    Args:
        width (int, optional): The width of the pride flag. If none is specified,
            the full width of the terminal is used.
    """
    if width is None:
        width = get_terminal_size(default_width=10).columns
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
    arr = np.repeat(np.linspace(0, 1, 6).reshape(-1, 1), width, axis=1)
    viz(arr, colorscheme=pride_cs, legend=False)
