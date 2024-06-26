import pytest
import jax.numpy as jnp
from jax import core
import numpy as np
from tensorhue.connectors._jax import _tensorhue_to_numpy_jax


class NonConvertibleTensor:
    pass


def test_jax_device_array():
    data = [[1, 2], [3, 4]]
    device_array = jnp.array(data)
    assert np.array_equal(_tensorhue_to_numpy_jax(device_array), np.array(data))


def test_tensor_dtypes():
    dtypes = {
        jnp.float32: "float32",
        jnp.bfloat16: "bfloat16",
        jnp.int32: "int32",
        jnp.uint8: "uint8",
        bool: "bool",
        jnp.complex64: "complex64",
    }
    jnp_array = jnp.array([0.0, 1.0, 2.0, jnp.nan, jnp.inf])
    for dtype_jnp, dtype_np in dtypes.items():
        jnp_casted = jnp_array.astype(dtype_jnp)
        converted = _tensorhue_to_numpy_jax(jnp_casted)
        assert np.array_equal(
            converted.dtype, dtype_np
        ), f"dtype mismatch in jax.numpy to numpy conversion: expected {dtype_np}, got {converted.dtype}"


def test_jax_incompatible_arrays():
    shape = (2, 2)
    dtype = jnp.float32

    shaped_array = core.ShapedArray(shape, dtype)
    with pytest.raises(NotImplementedError) as exc_info:
        _tensorhue_to_numpy_jax(shaped_array)
    assert "cannot be visualized" in str(exc_info.value)


def test_runtime_error_for_non_convertible_tensor():
    non_convertible = NonConvertibleTensor()
    with pytest.raises(NotImplementedError) as exc_info:
        _tensorhue_to_numpy_jax(non_convertible)
    assert "Got non-visualizable dtype 'object'." in str(exc_info.value)
