import numpy as np
import tensorflow as tf
import pytest
from tensorhue.connectors._tensorflow import _tensorhue_to_numpy_tensorflow


class NonConvertibleTensor:
    def numpy(self):
        raise RuntimeError("This tensor cannot be converted to numpy")


def test_tensor_dtypes():
    dtypes = {
        tf.float32: "float32",
        tf.double: "float64",
        tf.int32: "int32",
        tf.int64: "int64",
        tf.bool: "bool",
        tf.complex128: "complex128",
    }
    tf_tensor = tf.constant([0.0, 1.0, 2.0, float("nan"), float("inf")])
    for dtype_tf, dtype_np in dtypes.items():
        tensor_casted = tf.cast(tf_tensor, dtype_tf)
        converted = _tensorhue_to_numpy_tensorflow(tensor_casted)
        assert np.array_equal(
            converted.dtype, dtype_np
        ), f"dtype mismatch in torch to numpy conversion: expected {dtype_np}, got {converted.dtype}"


def test_runtime_error_for_non_convertible_tensor():
    non_convertible = NonConvertibleTensor()
    with pytest.raises(NotImplementedError) as exc_info:
        _tensorhue_to_numpy_tensorflow(non_convertible)
    assert "This tensor cannot be converted to numpy" in str(exc_info.value)


def test_unexpected_exception_for_other_errors():
    class UnexpectedErrorTensor:
        def numpy(self):
            raise ValueError("Unexpected error")

    with pytest.raises(RuntimeError) as exc_info:
        _tensorhue_to_numpy_tensorflow(UnexpectedErrorTensor())
    assert "Unexpected error" in str(exc_info.value)
