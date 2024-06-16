from rich.console import Console
import numpy as np
from tensorhue.colors import ColorScheme
from tensorhue._print_opts import PRINT_OPTS
from tensorhue.numpy import NumpyArrayWrapper


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


def _viz(self, colorscheme: ColorScheme = None):
    """
    Prints a tensor using colored Unicode art representation.

    Args:
        colorscheme (ColorScheme, optional): The color scheme to use.
            Defaults to None, which means the global default color scheme is used.
    """
    if colorscheme is None:
        colorscheme = PRINT_OPTS.colorscheme

    self = self._tensorhue_to_numpy()
    shape = self.shape

    if len(shape) > 2:
        raise NotImplementedError(
            "Visualization for tensors with more than 2 dimensions is under development. Please slice them for now."
        )

    colors = colorscheme(self)[..., :3]

    result_lines = [""]
    for y in range(0, shape[0] - 1, 2):
        for x in range(shape[-1]):
            result_lines[
                -1
            ] += f"[rgb({colors[y, x, 0]},{colors[y, x, 1]},{colors[y, x, 2]}) on rgb({colors[y+1, x, 0]},{colors[y+1, x, 1]},{colors[y+1, x, 2]})]▀[/]"
        result_lines.append("")

    if shape[0] % 2 == 1:
        for x in range(shape[1]):
            result_lines[-1] += f"[rgb({colors[-1, x, 0]},{colors[-1, x, 1]},{colors[-1, x, 2]})]▀[/]"

    c = Console(log_path=False, record=False)
    c.print("\n".join(result_lines))
