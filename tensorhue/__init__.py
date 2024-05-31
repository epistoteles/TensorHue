import sys
from rich.console import Console
import numpy as np
from tensorhue.version import VERSION
from tensorhue.colors import COLORS
__version__ = VERSION

__all__ = [
    "set_printoptions",
    "setup"
]

def viz(self) -> None:
    if isinstance(self, torch.FloatTensor):
        _viz_Tensor(self, COLORS['default_dark'], COLORS['default_bright'])
    elif isinstance(self, torch.BoolTensor):
        _viz_Tensor(self, COLORS['false'], COLORS['true'])


def _viz_Tensor(self, colors: tuple[tuple[int], tuple[int]] = None) -> None:
    data = self.data.numpy()
    shape = data.shape
    color_a = np.array(color_a)
    color_b = np.array(color_b)
    color = ((1 - data[::2, :, None]) * color_a + data[::2, :, None] * color_b).astype(int)
    bgcolor = ((1 - data[1::2, :, None]) * color_a + data[1::2, :, None] * color_b).astype(int)
    
    result_parts = []
    for y in range(shape[0] // 2):
        for x in range(shape[1]):
            result_parts.append(f"[rgb({color[y, x, 0]},{color[y, x, 1]},{color[y, x, 2]}) on rgb({bgcolor[y, x, 0]},{bgcolor[y, x, 1]},{bgcolor[y, x, 2]})]▀[/]")
        result_parts.append("\n")
    
    c = Console(log_path=False, record=False)
    c.print(''.join(result_parts))


# automagically set up tensorhue
if 'torch' in sys.modules:
    torch = sys.modules['torch']
    setattr(torch.Tensor, "viz", viz)
