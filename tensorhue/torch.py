import torch
import numpy as np


def _tensorhue_to_numpy_torch(tensor: torch.Tensor) -> np.ndarray:
    if isinstance(tensor, torch.masked.MaskedTensor):
        return np.ma.masked_array(tensor.get_data(), torch.logical_not(tensor.get_mask()))
    try:
        return tensor.numpy()
    except RuntimeError as e:
        raise NotImplementedError(
            f"{e}: It looks like tensors of type {type(tensor)} cannot be converted to numpy arrays out-of-the-box. Raise an issue if you need to visualize them."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while converting tensor of type {type(tensor)} to numpy array: {e}"
        ) from e
