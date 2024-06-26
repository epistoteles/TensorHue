import numpy as np


def _tensorhue_to_numpy_torch(tensor) -> np.ndarray:
    if tensor.__class__.__name__ == "MaskedTensor":  # hacky - but we shouldn't import torch here
        return np.ma.masked_array(tensor.get_data(), ~tensor.get_mask())
    try:  # pylint: disable=duplicate-code
        return tensor.numpy()
    except RuntimeError as e:
        raise NotImplementedError(
            f"{e}: It looks like tensors of type {type(tensor)} cannot be converted to numpy arrays out-of-the-box. Raise an issue if you need to visualize them."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while converting tensor of type {type(tensor)} to numpy array: {e}"
        ) from e
