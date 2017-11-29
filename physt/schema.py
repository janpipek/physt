import numpy as np
from typing import Tuple


class Schema:
    """

    Note:
    """
    def copy(self) -> 'Schema':
        raise NotImplementedError()

    @property
    def bins(self) -> np.ndarray:
        return self._bins

    @property
    def bins_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        return

    def fit(self, data):
        raise NotImplementedError()

    def fit_and_apply(self, data, weights=None) -> np.ndarray:
        # TODO: Handle data to make them 1D array
        data = np.asarray(data)
        self.fit(data)
        bins, mask = self.bins_and_mask
        # TODO: guess dtype
        numpy_result, _ = np.histogram(data, bins=bins, weights=weights)
        return numpy_result[mask].copy()


class StaticSchema(Schema):
    def __init__(self, bins):
        # TODO: Check bin property
        self._bins = bins.copy()

    def fit(self, data):
        # TODO: Warning
        pass

    def copy(self) -> 'StaticSchema':
        return self.__class__(self.bins)


class NumpySchema(Schema):
    def __init__(self, bins):
        self.bin_arg = bins

    def fit(self, data):
        _, self._bins = np.histogram(data, bins=self.bin_arg)

    def fit_and_apply(self, data, weights=None):
        numpy_result, self._bins = np.histogram(data, bins=self.bin_arg, weights=weights)
        return numpy_result