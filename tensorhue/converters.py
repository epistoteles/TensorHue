from __future__ import annotations
import warnings
import numpy as np


def tensor_to_numpy(tensor, **kwargs) -> np.ndarray:
    """
    Converts a tensor of unknown type to a numpy array.

    Args:
        tensor (Any): The tensor to be converted.
        **kwargs: Additional keyword arguments that are passed to the underlying converter functions.

    Returns:
        The converted numpy array.
    """
    mro_strings = mro_to_strings(tensor.__class__.__mro__)

    if "numpy.ndarray" in mro_strings:
        return tensor
    if "torch.Tensor" in mro_strings:
        return _tensor_to_numpy_torch(tensor, **kwargs)
    if "tensorflow.python.types.core.Tensor" in mro_strings:
        return _tensor_to_numpy_tensorflow(tensor, **kwargs)
    if "jaxlib.xla_extension.DeviceArray" in mro_strings:
        return _tensor_to_numpy_jax(tensor, **kwargs)
    if "PIL.Image.Image" in mro_strings:
        return _tensor_to_numpy_pillow(tensor, **kwargs)
    raise NotImplementedError(
        f"Conversion of tensor of type {type(tensor)} is not supported. Please raise an issue of you think this is a bug or should be implemented."
    )


def mro_to_strings(mro) -> list[str]:
    """
    Converts the __mro__ of a class to a list of module.class_name strings.

    Args:
        mro (tuple[type]): The __mro__ to be converted.

    Returns:
        The converted list of strings.
    """
    return [f"{c.__module__}.{c.__name__}" for c in mro]


def _tensor_to_numpy_torch(tensor) -> np.ndarray:
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


def _tensor_to_numpy_tensorflow(tensor) -> np.ndarray:
    if tensor.__class__.__name__ == "RaggedTensor":  # hacky - but we shouldn't import torch here
        warnings.warn(
            "Tensorflow RaggedTensors are currently converted to dense tensors by filling with the value 0. Values that are actually 0 and filled-in values will appear indistinguishable. This behavior will change in the future."
        )
        return _tensor_to_numpy_tensorflow(tensor.to_tensor())
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


def _tensor_to_numpy_jax(tensor) -> np.ndarray:
    not_implemented = {"ShapedArray", "UnshapedArray", "AbstractArray"}
    if {c.__name__ for c in tensor.__class__.__mro__}.intersection(
        not_implemented
    ):  # hacky - but we shouldn't import jax here
        raise NotImplementedError(
            f"Jax arrays of type {tensor.__class__.__name__} cannot be visualized. Raise an issue if you believe this is wrong."
        )
    try:
        array = np.asarray(tensor)
        if array.dtype == "object":
            raise RuntimeError("Got non-visualizable dtype 'object'.")
        return array
    except RuntimeError as e:
        raise NotImplementedError(
            f"{e}: It looks like JAX arrays of type {type(tensor)} cannot be converted to numpy arrays out-of-the-box. Raise an issue if you need to visualize them."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while converting tensor of type {type(tensor)} to numpy array: {e}"
        ) from e


def _tensor_to_numpy_pillow(image, thumbnail, max_size) -> np.ndarray:
    try:
        image = image.convert("RGB")
    except Exception as e:
        raise ValueError("Could not convert image from mode '{mode}' to 'RGB'.") from e

    if thumbnail:
        image.thumbnail(max_size)

    array = np.array(image)
    assert array.dtype == "uint8"

    return array
