import numpy as np
import torch
import pytest
from tensorhue.connectors._torch import _tensorhue_to_numpy_torch


class NonConvertibleTensor(torch.Tensor):
    def numpy(self):
        raise RuntimeError("This tensor cannot be converted to numpy")


@pytest.mark.filterwarnings("ignore::UserWarning:torch")
def test_masked_tensor():
    ones = torch.ones(5, 5)
    mask = torch.eye(5).bool()
    masked_torch = torch.masked.MaskedTensor(ones, mask)
    masked_numpy = np.ma.masked_array(ones.numpy(), ~mask.numpy())
    converted = _tensorhue_to_numpy_torch(masked_torch)
    assert np.array_equal(
        converted, masked_numpy
    ), "Converting masked tensor to masked array failed. Please check if the torch.masked API changed and raise an issue."


def test_tensor_dtypes():
    dtypes = {
        torch.FloatTensor: "float32",
        torch.DoubleTensor: "float64",
        torch.IntTensor: "int32",
        torch.LongTensor: "int64",
        torch.bool: "bool",
        torch.complex128: "complex128",
    }
    torch_tensor = torch.Tensor([0.0, 1.0, 2.0, torch.nan, torch.inf])
    for dtype_torch, dtype_np in dtypes.items():
        torch_casted = torch_tensor.type(dtype_torch)
        converted = _tensorhue_to_numpy_torch(torch_casted)
        assert np.array_equal(
            converted.dtype, dtype_np
        ), f"dtype mismatch in torch to numpy conversion: expected {dtype_np}, got {converted.dtype}"


def test_runtime_error_for_non_convertible_tensor():
    non_convertible = NonConvertibleTensor()
    with pytest.raises(NotImplementedError) as exc_info:
        _tensorhue_to_numpy_torch(non_convertible)
    assert "This tensor cannot be converted to numpy" in str(exc_info.value)


def test_unexpected_exception_for_other_errors():
    class UnexpectedErrorTensor:
        def numpy(self):
            raise ValueError("Unexpected error")

    with pytest.raises(RuntimeError) as exc_info:
        _tensorhue_to_numpy_torch(UnexpectedErrorTensor())
    assert "Unexpected error" in str(exc_info.value)
