import numpy as np

from .schema import Schema

__all__ = ["Histogram",]


class Histogram:
    """

    Attributes
    ----------
    - values
    - bins
    
    """
    def __init__(self, schema: Schema, values: np.ndarray):
        # self.dtype = int
        self._values = values
        self._schema = schema

    @property
    def schema(self):
        return self._schema

    @property
    def bins(self) -> np.ndarray:
        return self._schema.bins

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def total(self) -> int:
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
            If True, the values are not copied, but shared.
        """
        new_schema = self._schema.copy()
        new_values = self._values if shallow else self._values.copy()

        return self.__class__(schema=new_schema, values=new_values)

    def normalize(self, inplace=False) -> 'Histogram':
        if not self.values:
            raise RuntimeError("Cannot normalize histogram without values.")
        if not inplace:
            copy = self.copy(shallow=True)
            return copy.normalize(inplace=True)
        else:
            # TODO: Make sure to convert to float
            self._values /= self.total
            return self
    
    @property
    def densities(self) -> np.ndarray:
        pass
