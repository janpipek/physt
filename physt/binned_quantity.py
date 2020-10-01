from copy import deepcopy
from typing import List, Optional, Iterable, Tuple

import numpy as np

from physt.binnings import as_binning, BinningBase
from physt.typing_aliases import ArrayLike, BinningLike, DtypeLike, Axis, MetaData


class BinnedQuantity:
    binnings: List[BinningBase]
    meta_data: MetaData
    values: np.ndarray
    errors2: Optional[np.ndarray] = None

    @classmethod
    def __init__(
        self,
        binnings: Iterable[BinningLike],
        values: ArrayLike,
        *,
        dtype: Optional[DtypeLike] = None,
        errors2: Optional[ArrayLike] = None,
        axis_names: Optional[Iterable[str]],
        **kwargs
    ):
        self.binnings = [as_binning(binning) for binning in binnings]
        self.meta_data = kwargs
        self.meta_data["axis_names"] = list(axis_names) or self.default_axis_names
        if self.shape != values.shape:
            raise ValueError("Values must have same dimension as bins.")
        self.values = np.asarray(values, dtype=dtype)
        if errors2:
            self.errors2 = np.asarray(errors2)

    @property
    def bins(self) -> List[np.ndarray]:
        """List of bin matrices."""
        return [binning.bins for binning in self.binnings]

    @property
    def edges(self) -> List[np.ndarray]:
        return [binning.numpy_bins for binning in self.binnings]

    def numpy_bins(self) -> List[np.ndarray]:
        """Numpy-like bins (if available).

        TODO: Deprecate.
        """
        return self.edges

    def numpy_like(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Same result as would the numpy.histogram function return."""
        return self.values, self.edges

    def copy(self) -> "BinnedQuantity":
        a_copy = self.__class__.__new__(self.__class__)
        a_copy.binnings = [binning.copy() for binning in self.binnings]
        a_copy.values = np.copy(self.values)
        a_copy.meta_data = deepcopy(self.meta_data)
        a_copy.errors2 = np.copy(self.values) if self.values is not None else None
        return a_copy

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        Tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self.binnings)

    @property
    def ndim(self) -> int:
        """Dimensionality of histogram's data.

        i.e. the number of axes along which we bin the values.
        """
        return len(self.binnings)

    @property
    def axis_names(self) -> Tuple[str, ...]:
        """Names of axes (stored in meta-data)."""
        default = ["axis{0}".format(i) for i in range(self.ndim)]
        return tuple(self.meta_data.get("axis_names", None) or default)

    @axis_names.setter
    def axis_names(self, value: Iterable[str]):
        self.meta_data["axis_names"] = tuple(str(name) for name in value)

    @property
    def default_axis_names(self) -> List[str]:
        """Axis names that are used if not specified in the constructor."""
        return ["axis{0}".format(i) for i in range(self.ndim)]

    def _get_axis(self, name_or_index: Axis) -> int:
        """Get a zero-based index of an axis and check its existence."""
        # TODO: Add unit test
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= self.ndim:
                raise ValueError("No such axis, must be from 0 to {0}".format(self.ndim - 1))
            return name_or_index
        elif isinstance(name_or_index, str):
            if name_or_index not in self.axis_names:
                named_axes = [name for name in self.axis_names if name]
                raise ValueError(
                    "No axis with such name: {0}, available names: {1}. In most places, you can also use numbers.".format(
                        name_or_index, ", ".join(named_axes)
                    )
                )
            return self.axis_names.index(name_or_index)
        else:
            raise TypeError("Argument of type {0} not understood, int or str expected.".format(type(name_or_index)))

    def __array__(self) -> np.ndarray:
        """Convert to numpy array.

        Returns
        -------
        The values

        See also
        --------
        frequencies
        """
        return self.values


def have_same_bins(left: BinnedQuantity, right: BinnedQuantity) -> bool:
    if left.shape != right.shape:
        return False
    return np.allclose(left.bins, right.bins)


def merge_meta_data(left: BinnedQuantity, right: BinnedQuantity) -> MetaData:
    """Merge meta data of two quantities leaving only those where there is no conflict.

    (Used mostly in arithmetic operations).
    """
    keys = set(left.meta_data.keys())
    keys = keys.union(set(right.meta_data.keys()))
    return {
        key: (left.meta_data[key] if left.meta_data.get(key, None) == right.meta_data.get(key, None) else None)
        for key in keys
    }

