import numpy as np


class NumpyArrayWrapper(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def _tensorhue_to_numpy(self):
        return self
