import warnings
import numpy as np


def _tensorhue_to_numpy_tensorflow(tensor) -> np.ndarray:
    if tensor.__class__.__name__ == "RaggedTensor":  # hacky - but we shouldn't import torch here
        warnings.warn(
            "Tensorflow RaggedTensors are currently converted to dense tensors by filling with the value 0. Values that are actually 0 and filled-in values will appear indistinguishable. This behavior will change in the future."
        )
        return _tensorhue_to_numpy_tensorflow(tensor.to_tensor())
    if tensor.__class__.__name__ == "SparseTensor":
        raise ValueError("Tensorflow SparseTensors are not yet supported by TensorHue.")
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
