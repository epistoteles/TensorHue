import os
import sys
import warnings
from rich.console import Console
import numpy as np
from tensorhue.colors import ColorScheme
from tensorhue._print_opts import PRINT_OPTS
from tensorhue.connectors._numpy import NumpyArrayWrapper


def viz(tensor, **kwargs):
    if isinstance(tensor, np.ndarray):
        tensor = NumpyArrayWrapper(tensor)
        tensor.viz(**kwargs)  # pylint: disable=no-member
    else:
        try:
            tensor.viz(**kwargs)
        except Exception as e:
            raise NotImplementedError(
                f"TensorHue currently does not support type {type(tensor)}. Please raise an issue if you want to visualize them. Alternatively, check if you imported tensorhue *after* your other library."
            ) from e


def _viz(self, colorscheme: ColorScheme = None, legend: bool = True, scale: int = 1, **kwargs):
    """
    Prints a tensor using colored Unicode art representation.

    Args:
        colorscheme (ColorScheme, optional): The color scheme to use.
            Defaults to None, which means the global default color scheme is used.
        legend (bool, optional): Whether or not to include legend information (like the shape)
        scale (int, optional): Scales the size of the entire tensor up, making the unicode 'pixels' larger.
        **kwargs: Additional keyword arguments that are passed to the underlying viz function (vmin or vmax)
    """
    if not isinstance(scale, int):
        raise ValueError("scale must be an integer.")

    if colorscheme is None:
        colorscheme = PRINT_OPTS.colorscheme

    self = self._tensorhue_to_numpy()
    shape = self.shape
    ndim = self.ndim

    if ndim == 1:
        self = self[np.newaxis, :]
    elif ndim > 2:
        raise NotImplementedError(
            "Visualization of tensors with more than 2 dimensions is under development. Please slice them for now."
        )

    self = np.repeat(np.repeat(self, scale, axis=1), scale, axis=0)

    result_lines = _viz_2d(self, colorscheme, **kwargs)

    if legend:
        result_lines.append(f"[italic]shape = {shape}[/]")

    c = Console(log_path=False, record=False)
    c.print("\n".join(result_lines))


def _viz_2d(array_2d: np.ndarray, colorscheme: ColorScheme = None, **kwargs) -> list[str]:
    """
    Constructs a list of rich-compatible strings out of a 2D numpy array.

    Args:
        array_2d (np.ndarray): The 2-dimensional numpy array (or 3-dimensional if the values are already RGB).
        colorscheme (ColorScheme): The color scheme to use. If None, the array must be 3-dimensional (already RGB values).
        **kwargs: Additional keyword arguments that are passed to the underlying viz function (vmin or vmax)
    """
    terminal_width = get_terminal_size().columns
    shape = array_2d.shape

    if shape[1] > terminal_width:
        slice_left = (terminal_width - 5) // 2
        slice_right = slice_left + (terminal_width - 5) % 2
        if colorscheme is not None:
            colors_right = colorscheme(array_2d[:, -slice_right:])[..., :3]
        else:
            assert (
                array_2d.ndim == 3 and array_2d.shape[-1] == 3
            ), "Array shape must be 3-dimensional (*, *, 3) when colorscheme=None."
            colors_right = array_2d[:, -slice_right:, :]
    else:
        slice_left = shape[1]
        slice_right = colors_right = False

    if colorscheme is not None:
        colors_left = colorscheme(array_2d[:, :slice_left], **kwargs)[..., :3]
    else:
        assert (
            array_2d.ndim == 3 and array_2d.shape[-1] == 3
        ), "Array shape must be 3-dimensional (*, *, 3) when colorscheme=None."
        colors_left = array_2d[:, :slice_left, :]

    result_lines = _construct_unicode_string(colors_left, colors_right)

    return result_lines


def _construct_unicode_string(colors_left: np.ndarray, colors_right: np.ndarray) -> str:
    result_lines = [""]

    for y in range(0, colors_left.shape[0] - 1, 2):
        for x in range(colors_left.shape[1]):
            result_lines[
                -1
            ] += f"[rgb({colors_left[y, x, 0]},{colors_left[y, x, 1]},{colors_left[y, x, 2]}) on rgb({colors_left[y+1, x, 0]},{colors_left[y+1, x, 1]},{colors_left[y+1, x, 2]})]▀[/]"
        if isinstance(colors_right, np.ndarray):
            result_lines[-1] += " ··· "
            for x in range(colors_right.shape[1]):
                result_lines[
                    -1
                ] += f"[rgb({colors_right[y, x, 0]},{colors_right[y, x, 1]},{colors_right[y, x, 2]}) on rgb({colors_right[y+1, x, 0]},{colors_right[y+1, x, 1]},{colors_right[y+1, x, 2]})]▀[/]"
        result_lines.append("")

    if colors_left.shape[0] % 2 == 1:
        for x in range(colors_left.shape[1]):
            result_lines[-1] += f"[rgb({colors_left[-1, x, 0]},{colors_left[-1, x, 1]},{colors_left[-1, x, 2]})]▀[/]"
        if isinstance(colors_right, np.ndarray):
            result_lines[-1] += " ··· "
            for x in range(colors_right.shape[1]):
                result_lines[
                    -1
                ] += f"[rgb({colors_right[-1, x, 0]},{colors_right[-1, x, 1]},{colors_right[-1, x, 2]})]▀[/]"
    else:
        result_lines = result_lines[:-1]

    return result_lines


def _viz_image(self, legend: bool = False, thumbnail: bool = True, max_size: tuple[int, int] = None):
    """
    A special case of _viz that does not use the ColorScheme but instead treats the tensor as RGB or greyscale values directly.

    Args:
        legend (bool, optional): Whether or not to include legend information (like the shape)
        thumbnail (bool, optional): Scales down the image size to a thumbnail that fits into the terminal window
        max_size (tuple[int, int], optional): The maximum size (width, height) to which the image gets downsized to. Only used if thumbnail=True.
    """

    raise_max_size_warning = max_size and not thumbnail

    size = self.size
    mode = self.mode
    if max_size is None:
        terminal_size = get_terminal_size()
    else:
        terminal_size = os.terminal_size(max_size)
    max_size = (terminal_size.columns, (terminal_size.lines - 1) * 2)
    self = self._tensorhue_to_numpy(thumbnail=thumbnail, max_size=max_size)

    result_lines = _viz_2d(self)

    if legend:
        result_lines.append(f"[italic]size = {size}[/], [italic]mode = {mode}[/]")

    c = Console(log_path=False, record=False)
    c.print("\n".join(result_lines))

    if raise_max_size_warning:
        warnings.warn(
            "You specified a max_size, but set thumbnail to False. Your max_size will be ignored unless thumbnail=True."
        )


def get_terminal_size(default_width: int = 100, default_height: int = 70) -> os.terminal_size:
    """
    Returns the terminal size if the standard output is connected to a terminal. Otherwise, returns the defined default size.

    Args:
        default_width (int, optional): The default width to use if there is no terminal connected.
        default_height (int, optional): The default height to use if there is no terminal connected.
    """
    if sys.stdout.isatty():
        try:
            return os.get_terminal_size()
        except OSError:
            return os.terminal_size((default_width, default_height))
    else:
        return os.terminal_size((default_width, default_height))
