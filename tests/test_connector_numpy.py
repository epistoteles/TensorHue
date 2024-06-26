import numpy as np
from tensorhue.connectors._numpy import NumpyArrayWrapper


def test_instantiation():
    data = [1, 2, 3]
    wrapped = NumpyArrayWrapper(data)
    assert isinstance(wrapped, NumpyArrayWrapper)
    assert isinstance(wrapped, np.ndarray)
    assert np.array_equal(wrapped, np.array(data))


def test_functionality_preservation():
    wrapped = NumpyArrayWrapper([1, 2, 3])
    assert wrapped.sum() == 6
    assert isinstance(wrapped * 2, NumpyArrayWrapper)
    assert np.array_equal(wrapped[1:], np.array([2, 3]))
    assert isinstance(np.sqrt(wrapped), NumpyArrayWrapper)


def test_tensorhue_to_numpy():
    wrapped = NumpyArrayWrapper([4, 5, 6])
    result = wrapped._tensorhue_to_numpy()
    assert result is wrapped
