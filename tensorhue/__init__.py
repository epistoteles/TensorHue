import sys
from rich.console import Console
import numpy as np
from tensorhue.colors import COLORS

__version__ = "0.0.2"  # single source of version truth

__all__ = []


def viz(self) -> None:
    """
    Prints the tensor using a colored Unicode art representation.
    This method checks the type of the tensor and calls the `_viz_tensor` function with the appropriate colors.
    """
    if isinstance(self, torch.FloatTensor):  # pylint: disable=possibly-used-before-assignment
        _viz_tensor(self)
    elif isinstance(self, torch.BoolTensor):
        _viz_tensor(self, (COLORS["false"], COLORS["true"]))


def _viz_tensor(self, colors: tuple[tuple[int], tuple[int]] = None) -> None:
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
    if colors is None:
        colors = COLORS["default_dark"], COLORS["default_bright"]
    data = self.data.numpy()
    shape = data.shape
    color_a = np.array(colors[0])
    color_b = np.array(colors[1])
    color = ((1 - data[::2, :, None]) * color_a + data[::2, :, None] * color_b).astype(int)
    bgcolor = ((1 - data[1::2, :, None]) * color_a + data[1::2, :, None] * color_b).astype(int)

    result_parts = []
    for y in range(shape[0] // 2):
        for x in range(shape[1]):
            result_parts.append(
                f"[rgb({color[y, x, 0]},{color[y, x, 1]},{color[y, x, 2]}) on rgb({bgcolor[y, x, 0]},{bgcolor[y, x, 1]},{bgcolor[y, x, 2]})]▀[/]"
            )
        result_parts.append("\n")
    if shape[0] % 2 == 1:
        for x in range(shape[1]):
            result_parts.append(f"[rgb({color[-1, x, 0]},{color[-1, x, 1]},{color[-1, x, 2]})]▀[/]")

    c = Console(log_path=False, record=False)
    c.print("".join(result_parts))
    return "".join(result_parts)


# automagically set up TensorHue
if "torch" in sys.modules:
    torch = sys.modules["torch"]
    setattr(torch.Tensor, "viz", viz)


# def _viz_tensor_alt(self, colors: tuple[tuple[int], tuple[int]] = None) -> None:
#     if colors is None:
#         colors = COLORS["default_dark"], COLORS["default_bright"]
#     data = self.data.numpy()
#     shape = data.shape
#     dim = data.ndim
#     color_a = np.array(colors[0])
#     color_b = np.array(colors[1])
#     color = ((1 - data[::2, :, None]) * color_a + data[::2, :, None] * color_b).astype(int)
#     bgcolor = ((1 - data[1::2, :, None]) * color_a + data[1::2, :, None] * color_b).astype(int)

#     result_parts = []
#     for y in range(shape[0] // 2):
#         for x in range(shape[1]):
#             result_parts.append(
#                 f"[rgb({color[y, x, 0]},{color[y, x, 1]},{color[y, x, 2]}) on rgb({bgcolor[y, x, 0]},{bgcolor[y, x, 1]},{bgcolor[y, x, 2]})]▀[/]"
#             )
#         result_parts.append("\n")
#     if shape[0] % 2 == 1:
#         for x in range(shape[1]):
#             result_parts.append(f"[rgb({color[-1, x, 0]},{color[-1, x, 1]},{color[-1, x, 2]})]▀[/]")

#     c = Console(log_path=False, record=False)
#     c.print("".join(result_parts))
#     # return "".join(result_parts)
