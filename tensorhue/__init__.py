import sys
from rich.console import Console
import numpy as np
from tensorhue.colors import COLORS, ColorScheme
from tensorhue._print_opts import PRINT_OPTS, set_printoptions


__version__ = "0.0.2"  # single source of version truth

__all__ = ["set_printoptions"]


# def viz(self) -> None:
#     """
#     Prints the tensor using a colored Unicode art representation.
#     This method checks the type of the tensor and calls the `_viz_tensor` function with the appropriate colors.
#     """
#     if isinstance(self, (torch.FloatTensor, torch.IntTensor, torch.LongTensor)):  # pylint: disable=possibly-used-before-assignment
#         _viz_tensor(self)
#     elif isinstance(self, torch.BoolTensor):
#         _viz_tensor(self, (COLORS["false"], COLORS["true"]))


def viz(self, colorscheme: ColorScheme = None) -> None:
    """
    Prints a tensor using colored Unicode art representation.

    This function takes a tensor and a tuple of two tuples of integers representing the colors.
    It converts the tensor data to a numpy array, calculates the colors for each element based on the input colors,
    and generates a string representation of the tensor using the calculated colors.
    The resulting string is then printed using the Console class from the rich library.

    Parameters:
        colors (tuple[tuple[int], tuple[int]]): A tuple of two RGB tuples representing the colors.
            The first tuple represents the RGB color for the smallest value in the tensor, the second tuple represents
            the RGB color for the biggest value of the tensor. (Default = None; uses default colors)
    """
    if colorscheme is None:
        colorscheme = PRINT_OPTS.colorscheme

    data = self.data.numpy()
    shape = data.shape

    colors = colorscheme(data)[..., :3]

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


# automagically set up TensorHue
if "torch" in sys.modules:
    torch = sys.modules["torch"]
    setattr(torch.Tensor, "viz", viz)
