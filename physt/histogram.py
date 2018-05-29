from typing import Tuple
from collections.abc import Mapping
from collections import UserDict
from numbers import Number

import numpy as np

from .schema import Schema
from .histogram_meta_data import HistogramMetaData

__all__ = ["Histogram",]


class Histogram:


    """

    Attributes
    ----------
    - values
    - bins
    
    """
    def __init__(self, schema: Schema, values: np.ndarray, meta_data: Mapping=None):
        self._values = values
        self._schema = schema
        self._dtype = None
        self._meta_data = HistogramMetaData(meta_data)

    @property
    def meta_data(self):
        return self._meta_data

    @property
    def dtype(self):
        # TODO: Rethink
        if self._values is not None:
            return self._values.dtype
        else:
            return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._values = self._values.astype(value)

    @property
    def ndim(self) -> int:
        return self._schema.ndim

    @property
    def shape(self) -> Tuple[int]:
        return self._schema.shape

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def bins(self) -> np.ndarray:
        return self._schema.bins

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def total(self) -> Number:
        if self.values is None:
            return None
        return self.values.sum()

    def __repr__(self) -> str:
        return("{0}[{1}](shape={2}, total={3})".format(
            self.__class__.__name__, self._schema.ndim, self._schema.shape, self.total
        ))

    def __array__(self) -> np.ndarray:
        """Convert to numpy array.

        Returns
        -------
        np.ndarray
            The array of frequencies

        See also
        --------
        frequencies
        """
        return self.values

    def copy(self, shallow:bool=False) -> 'Histogram':
        """"Create an identical copy of the histogram.
        
        Parameters
        ----------
        shallow:
            If True, the values are not copied, but shared
            (useful when the copy is immediately discarded).
        """
        new_schema = self._schema.copy()
        new_values = self._values if shallow else self._values.copy()

        return self.__class__(schema=new_schema, values=new_values)

    def normalize(self, inplace=False) -> 'Histogram':
        if self.values is None:
            raise RuntimeError("Cannot normalize histogram without values.")
        if not inplace:
            copy = self.copy(shallow=True)
            return copy.normalize(inplace=True)
        else:
            # TODO: Make sure to convert to float
            self.dtype = np.float
            self._values /= self.total
            return self

    @property
    def densities(self) -> np.ndarray:
        if self._values is None:
            return None
        else:
            return self._values / self._schema.bin_sizes


# Clean up namespace
del Tuple, Mapping, UserDict
