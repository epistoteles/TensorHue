import numpy as np


def _tensorhue_to_numpy_jax(tensor) -> np.ndarray:
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
