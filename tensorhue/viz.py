import os
import sys
from rich.console import Console
import numpy as np
from tensorhue.colors import ColorScheme
from tensorhue._print_opts import PRINT_OPTS
from tensorhue._numpy import NumpyArrayWrapper


def viz(tensor, *args, **kwargs):
    if isinstance(tensor, np.ndarray):
        tensor = NumpyArrayWrapper(tensor)
        tensor.viz(*args, **kwargs)  # pylint: disable=no-member
    else:
        try:
            tensor.viz(*args, **kwargs)
        except Exception as e:
            raise NotImplementedError(
                f"TensorHue does not support type {type(tensor)}. Raise an issue if you need to visualize them. Alternatively, check if you imported tensorhue *after* your other library."
            ) from e


def _viz(self, colorscheme: ColorScheme = None, legend: bool = True):
    """
    Prints a tensor using colored Unicode art representation.

    Args:
        colorscheme (ColorScheme, optional): The color scheme to use.
            Defaults to None, which means the global default color scheme is used.
        legend (bool, optional): Whether or not to include legend information (like the shape)
    """
    if colorscheme is None:
        colorscheme = PRINT_OPTS.colorscheme

    self = self._tensorhue_to_numpy()
    shape = self.shape

    if len(shape) == 1:
        self = self[np.newaxis, :]
    elif len(shape) > 2:
        raise NotImplementedError(
            "Visualization for tensors with more than 2 dimensions is under development. Please slice them for now."
        )

    result_lines = _viz_2d(self, colorscheme)

    if legend:
        result_lines.append(f"[italic]shape = {shape}[/]")

    c = Console(log_path=False, record=False)
    c.print("\n".join(result_lines))


def _viz_2d(array_2d: np.ndarray, colorscheme: ColorScheme) -> list[str]:
    """
    Constructs a list of rich-compatible strings out of a 2D numpy array.

    Args:
        array_2d (np.ndarray): The 2-dimensional numpy array
        colorscheme (ColorScheme): The color scheme to use
    """
    result_lines = [""]
    terminal_width = get_terminal_width()
    shape = array_2d.shape

    if shape[1] > terminal_width:
        slice_left = (terminal_width - 5) // 2
        slice_right = slice_left + (terminal_width - 5) % 2
        colors_right = colorscheme(array_2d[:, -slice_right:])[..., :3]
    else:
        slice_left = shape[1]
        slice_right = colors_right = False

    colors_left = colorscheme(array_2d[:, :slice_left])[..., :3]

    for y in range(0, shape[0] - 1, 2):
        for x in range(slice_left):
            result_lines[
                -1
            ] += f"[rgb({colors_left[y, x, 0]},{colors_left[y, x, 1]},{colors_left[y, x, 2]}) on rgb({colors_left[y+1, x, 0]},{colors_left[y+1, x, 1]},{colors_left[y+1, x, 2]})]▀[/]"
        if slice_right:
            result_lines[-1] += " ··· "
            for x in range(slice_right):
                result_lines[
                    -1
                ] += f"[rgb({colors_right[y, x, 0]},{colors_right[y, x, 1]},{colors_right[y, x, 2]}) on rgb({colors_right[y+1, x, 0]},{colors_right[y+1, x, 1]},{colors_right[y+1, x, 2]})]▀[/]"
        result_lines.append("")

    if shape[0] % 2 == 1:
        for x in range(slice_left):
            result_lines[-1] += f"[rgb({colors_left[-1, x, 0]},{colors_left[-1, x, 1]},{colors_left[-1, x, 2]})]▀[/]"
        if slice_right:
            result_lines[-1] += " ··· "
            for x in range(slice_right):
                result_lines[
                    -1
                ] += f"[rgb({colors_right[-1, x, 0]},{colors_right[-1, x, 1]},{colors_right[-1, x, 2]})]▀[/]"
    else:
        result_lines = result_lines[:-1]

    return result_lines


def get_terminal_width(default_width: int = 100) -> int:
    """
    Returns the terminal width if the standard output is connected to a terminal. Otherwise, returns default_width.

    Args:
        default_width (int, optional): The default width to use if there is no terminal.
    """
    if sys.stdout.isatty():
        try:
            return os.get_terminal_size().columns
        except OSError:
            return default_width
    else:
        return default_width
