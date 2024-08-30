"""Functions for individual steps of histogram and binning creation."""
import warnings
from functools import singledispatch
from typing import Any, Iterable, Iterator, List, Optional, Tuple, cast, overload

import numpy as np

from physt import _bin_utils
from physt.binnings import (
    BinningBase,
    bincount_methods,
    binning_methods,
    ideal_bin_count,
    numpy_binning,
    static_binning,
)
from physt.statistics import Statistics
from physt.typing_aliases import DTypeLike


@singledispatch
def extract_1d_array(
    data: Any, *, dropna: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract 1D array from any input.

    Parameters
    ----------
    data : Whatever array-like or iterable where each item is one observation.
        No shape is prescribed
    dropna : Whether to remove any NA values and construct the mask

    Returns
    -------
    array : None if no data, else 1-d float array (flattened)
    array_mask : None if no data or not required, else a boolean array
          to mark all indices of the original array that are non-na.

    Note
    ----
    To implement for another type, register via the singledispatch mechanism.
    """
    if isinstance(data, Iterator):
        # Numpy cannot iterators convert directly
        data = list(data)
    if np.isscalar(data):
        raise ValueError(f"Cannot extract array data from scalar {data!r}.")
    try:
        array: np.ndarray = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot extract array data from {type(data)}") from exc
    if dropna:
        array_mask = ~np.isnan(array)
        array = array[array_mask]
    else:
        array = array.flatten()
        array_mask = None
    return array, array_mask


@extract_1d_array.register
def _(data: None, *, dropna=True):
    return None, None


@singledispatch
def extract_nd_array(
    data: Any, *, dim: Optional[int] = None, dropna: bool = True
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract 2D tabular-like array from any input.

    Parameters
    ----------
    data : Whatever 2D array-like or iterable with rows as observations
        and columns for the dimension of the histogram
    dim : If no data, used to provide dim info, but can be also used to check data
    dropna : Whether to remove any NA values and construct the mask

    Returns
    -------
    dim : dimension of the data
    array :None if no data, else 2-d float array (dim columns, N rows)
    array_mask : None if no data or not required, else a boolean array
          to mark all indices of the original array that are non-na.

    Raises
    ------
    ValueError: If the dimensions don't agree

    Note
    ----
    To implement for another type, register via the singledispatch mechanism.
    """

    try:
        array: np.ndarray = np.asarray(data, dtype=float)
    except ValueError as exc:
        if "The requested array has an inhomogeneous shape" in str(exc):
            raise ValueError("Data must have a regular 2D shape of (n, d)") from exc
        raise
    if array.ndim != 2:
        raise ValueError(
            f"Data must have a 2D shape of (n, d), {array.shape} encountered."
        )
    if dim is not None and dim != array.shape[1]:
        raise ValueError(f"Dimension mismatch: {dim} != {array.shape[1]}")
    _, dim = array.shape
    if dropna:
        array_mask = ~np.isnan(array).any(axis=1)
        array = array[array_mask]
    else:
        array_mask = None
    return cast(int, dim), array, array_mask


@extract_nd_array.register
def _(data: None, *, dim=None, dropna=True):
    if dim is None:
        raise ValueError("You have to specify either data or its dimension.")
    if dim < 2:
        raise ValueError(f"Dimension too small: {dim}. At least 2 required.")
    return dim, None, None


def extract_and_concat_arrays(
    *data: Optional[Any], dropna: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Align multiple data arrays as columns into a single one.

    Parameters
    ----------
    data : Sequence of array-like or None's (not necessarily one-dimensional)
    dropna : Whether to remove any NA values and construct the mask

    Returns
    -------
    array : None or a 2D float array with rows as observations
    array_mask : None or a boolean array marking rows with non-na values

    Raises
    ------
    ValueError : If the dimensions do not match

    """
    none_count = sum(item is None for item in data)
    if none_count == len(data):
        return None, None
    if 1 <= none_count < len(data):
        raise ValueError(
            f"{none_count} None's on the input, 0 or {len(data)} expected."
        )
    array_list = [
        cast(np.ndarray, extract_1d_array(item, dropna=False)[0]) for item in data
    ]
    array_shapes = set(array.shape for array in array_list)
    if len(array_shapes) > 1:
        raise ValueError(f"Array shapes do not match: {list(array_shapes)}")
    array = np.concatenate([arr[:, np.newaxis] for arr in array_list], axis=1)
    if dropna:
        array_mask = ~np.isnan(array).any(axis=1)
        array = array[array_mask]
    else:
        array_mask = None
    return array, array_mask


@singledispatch
def extract_axis_name(data: Any, *, axis_name: Optional[str] = None) -> Optional[str]:
    """For input data, find the axis name (if there is any).

    Typically, this is the name of the data object (Series, ...)

    Parameters
    ----------
    data : Input data of the histogram
    axis_name : Explicitly set name that takes precedence

    Returns
    -------
    A name or None (no default set)

    Notes
    -----
    - To implement for another type, register via the singledispatch mechanism.
    - For data that are not handled by extract_1d_array, the result is not defined.

    """
    if not axis_name:
        if hasattr(data, "name"):
            return _normalize_axis_name(data.name)  # type: ignore
        elif (
            hasattr(data, "fields")
            and len(data.fields) == 1  # type: ignore
            and isinstance(data.fields[0], str)  # type: ignore
        ):
            # TODO: Move to dask or something
            # Case of dask fields (examples)
            return str(data.fields[0])  # type: ignore
    return axis_name


@singledispatch
def extract_axis_names(
    data: Any, *, axis_names: Optional[Iterable[str]] = None
) -> Optional[Tuple[str, ...]]:
    """For input data, find the names of the axes (if there are any).

    Typically, these are column names for dataframes etc.

    Parameters
    ----------
    data : Input data of the histogram
    axis_names : Explicitly set names that take precedence

    Returns
    -------
    A tuple of names or None's (no defaults provided)

    Notes
    -----
    - To implement for another type, register via the singledispatch mechanism.
    - For data that are not handled by extract_nd_array, the result is not defined.

    """
    if axis_names is not None:
        return tuple(axis_names)
    if hasattr(data, "columns"):
        return tuple(_normalize_axis_name(c) for c in data.columns)  # type: ignore
    return None


def _normalize_axis_name(axis_name: Any) -> Optional[str]:
    if axis_name is None:
        return None
    if isinstance(axis_name, (list, tuple)):
        return ", ".join(str(item) for item in axis_name)
    return str(axis_name)


@singledispatch
def extract_weights(
    weights: Any, *, array_mask: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Extract weights from the provided object.

    Returns
    -------
    weights: A 1d array with the values not in the mask removed.

    Note
    ----
    To implement for another type, register via the singledispatch mechanism.
    """
    # TODO: Improve docstring
    if weights is None:
        return None
    weights_array = np.asarray(weights)
    if array_mask is not None:
        if array_mask.shape != weights_array.shape:
            raise ValueError(
                f"Weights array shape ({weights_array.shape}) != expected ({array_mask.shape})."
            )
        weights_array = weights_array[array_mask]
    else:
        weights_array = weights_array.flatten()
    return weights_array


@overload
def calculate_nd_frequencies(
    data: np.ndarray,
    binnings: Iterable[BinningBase],
    weights: Optional[np.ndarray] = None,
    *,
    dtype: Optional[DTypeLike] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    ...


@overload
def calculate_nd_frequencies(
    data: None,
    binnings: Iterable[BinningBase],
    weights: Optional[np.ndarray] = None,
    *,
    dtype: Optional[DTypeLike] = None,
) -> Tuple[None, None, float]:
    ...


def calculate_nd_frequencies(
    data: Optional[np.ndarray],
    binnings: Iterable[BinningBase],
    weights: Optional[np.ndarray] = None,
    *,
    dtype: Optional[DTypeLike] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Get frequencies and bin errors from the data (n-dimensional variant).

    Parameters
    ----------
    data : 2D array with ndim columns and row for each entry.
    binnings: Binnings to apply in all axes.
    weights : 1D array of weights to assign to values.
        (If present, must have same length as the number of rows.)
    dtype : Underlying type for the histogram.
        (If weights are specified, default is float. Otherwise int64.)

    Returns
    -------
    frequencies : Frequencies (if data supplied)
    errors2 : Errors squared if different from frequencies
    missing : scalar[dtype]

    Raises
    ------
    ValueError
    """
    if data is None:
        return None, None, 0

    # Prepare numpy array of data
    if data.ndim != 2:
        raise ValueError(
            f"calculate_frequencies requires 2D input data, dim={data.ndim} found."
        )

    # Guess correct dtype and apply to weights
    if weights is None:
        if not dtype:
            dtype = np.int64
    else:
        if data is None:
            raise ValueError("Weights specified but data not.")
        if data.shape[0] != weights.shape[0]:
            raise ValueError("Different number of entries in data and weights.")
        if dtype:
            dtype = np.dtype(dtype)
            if dtype.kind in "iu" and weights.dtype.kind == "f":
                raise ValueError(
                    "Integer histogram requested but float weights entered."
                )
        else:
            dtype = weights.dtype

    edges_and_mask = [binning.numpy_bins_with_mask for binning in binnings]
    edges = [em[0] for em in edges_and_mask]
    masks = [em[1] for em in edges_and_mask]

    ixgrid = np.ix_(*masks)  # Indexer to select parts we want

    # TODO: Right edges are not taken into account because they fall into inf bin
    frequencies, _ = np.histogramdd(data, edges, weights=weights)
    frequencies = frequencies.astype(dtype)  # Automatically copy
    frequencies = frequencies[ixgrid]
    if weights is not None:
        missing = weights.sum() - frequencies.sum()
        err_freq, _ = np.histogramdd(data, edges, weights=weights**2)
        errors2 = err_freq[ixgrid].astype(dtype)  # Automatically copy
    else:
        missing = data.shape[0] - frequencies.sum()
        errors2 = None

    return frequencies, errors2, missing


def calculate_1d_frequencies(
    data: Optional[np.ndarray],
    binning: BinningBase,
    weights: Optional[np.ndarray] = None,
    *,
    validate_bins: bool = True,
    already_sorted: bool = False,
    dtype: Optional[DTypeLike] = None,
) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], float, float, Optional[Statistics]
]:
    """Get frequencies and bin errors from the data.

    Parameters
    ----------
    data : Data items to work on.
    binning : A set of bins.
    weights : Weights of the items.
    validate_bins : If True (default), bins are validated to be in ascending order.
    already_sorted : If True, the data being entered are already sorted, no need to sort them once more.
    dtype: Underlying type for the histogram.
        (If weights are specified, default is float. Otherwise long.)

    Returns
    -------
    frequencies : Bin contents
    errors2 :  Error squares of the bins
    underflow : Weight of items smaller than the first bin
    overflow : Weight of items larger than the last bin
    stats: The statistics (computed or empty)

    Note
    ----
    Checks that the bins are in a correct order (not necessarily consecutive).
    Does not check for numerical overflows in bins.
    """

    if data is None:
        return None, None, 0.0, 0.0, None

    # TODO: Is it possible to merge with histogram_nd.calculate_frequencies?

    underflow = np.nan
    overflow = np.nan

    # Ensure correct binning
    bins = binning.bins  # bin_utils.make_bin_array(bins)
    if validate_bins:
        if bins.shape[0] == 0:
            raise ValueError("Cannot have histogram with 0 bins.")
        if not _bin_utils.is_rising(bins):
            raise ValueError("Bins must be rising.")

    # Prepare 1D numpy array of data
    data_array = data
    if data_array.ndim > 1:
        # TODO: Perhaps disallow this?
        data_array = data_array.flatten()

    # Prepare 1D numpy array of weights
    if weights is not None:
        # TODO: It should be an array already
        weights_array = weights
        if weights_array.ndim > 1:
            weights_array = weights_array.flatten()

        # Check compatibility of weights
        if weights_array.shape != data_array.shape:
            raise ValueError(
                f"Weights must have the same shape as data, {weights_array.shape} != {data_array.shape}"
            )
        equal_weights = weights_array.max() - weights_array.min() == 0
    else:
        weights_array = np.ones_like(data_array, dtype=int)
        equal_weights = True

    # Prepare dtype
    inferred_dtype: np.dtype = np.dtype(dtype or weights_array.dtype)
    if inferred_dtype.kind in "iu" and weights_array.dtype.kind == "f":
        raise ValueError("Integer histogram requested but float weights entered.")

    # Data sorting
    if not already_sorted:
        sort_order = np.argsort(data_array)  # Memory: another copy
        data_array = data_array[sort_order]  # Memory: another copy
        weights_array = weights_array[sort_order]
        del sort_order

    # Fill frequencies and errors
    frequencies = np.zeros(bins.shape[0], dtype=inferred_dtype)
    errors2 = np.zeros(bins.shape[0], dtype=inferred_dtype)
    for xbin, bin in enumerate(bins):
        start = np.searchsorted(data_array, bin[0], side="left")
        stop = np.searchsorted(data_array, bin[1], side="left")

        if xbin == 0:
            underflow = weights_array[0:start].sum()
        if xbin == len(bins) - 1:
            stop = np.searchsorted(
                data_array, bin[1], side="right"
            )  # TODO: Understand and explain
            overflow = weights_array[stop:].sum()

        frequencies[xbin] = weights_array[start:stop].sum()
        errors2[xbin] = (weights_array[start:stop] ** 2).sum()

    # Underflow and overflow don't make sense for unconsecutive binning.
    if not _bin_utils.is_consecutive(bins):
        underflow = np.nan
        overflow = np.nan

    # Statistics
    if not data_array.size:
        stats = Statistics()
    else:
        stats = Statistics(
            sum=(data_array * weights_array).sum(),
            sum2=(data_array**2 * weights_array).sum(),
            min=float(data_array.min()),
            max=float(data_array.max()),
            weight=float(weights_array.sum()),
            # TODO: Support median with weights?
            median=np.median(data_array) if equal_weights else np.nan,
        )
    return frequencies, errors2, underflow, overflow, stats


def calculate_1d_bins(
    array: Optional[np.ndarray], _: Any = None, **kwargs
) -> BinningBase:
    """Find optimal binning from arguments.

    Parameters
    ----------
    array: Data from which the bins should be decided (sometimes used, sometimes not)
    _: int or str or Callable or arraylike or Iterable or BinningBase
        To-be-guessed parameter that specifies what kind of binning should be done
    check_nan: bool
        Check for the presence of nan's in array? Default: True
    range: Limit values to a range. Some binning methods also (subsequently) use this parameter for the bin shape.

    Returns
    -------
    BinningBase
        A two-dimensional array with pairs of bin edges (not necessarily consecutive).

    """
    if array is not None:
        if kwargs.pop("check_nan", True):
            if np.any(np.isnan(array)):
                raise ValueError("Cannot calculate bins in presence of NaN's.")
        if kwargs.get("range"):  # TODO: re-consider the usage of this parameter
            array = array[(array >= kwargs["range"][0]) & (array <= kwargs["range"][1])]
    if _ is None:
        bin_count = (
            10  # kwargs.pop("bins", ideal_bin_count(data=array)) - same as numpy
        )
        binning = numpy_binning(array, bin_count, **kwargs)
    elif isinstance(_, BinningBase):
        binning = _
    elif isinstance(_, int):
        binning = numpy_binning(array, _, **kwargs)
    elif isinstance(_, str):
        # What about the ranges???
        if _ in bincount_methods:
            # TODO: Do we really want this?
            if array is None:
                raise ValueError(
                    f"Cannot find the ideal number of bins without data (method='{_}')"
                )
            bin_count = ideal_bin_count(array, method=_)
            binning = numpy_binning(array, bin_count, **kwargs)
        elif _ in binning_methods:
            method = binning_methods[_]
            binning = method(array, **kwargs)
        else:
            raise ValueError(f"No binning method '{_}' available.")
    elif callable(_):
        binning = _(array, **kwargs)
    elif np.iterable(_):
        if isinstance(_, list):
            warnings.warn(
                "Using `list` for bins not recommended, it has different meaning with N-D histograms."
            )
        binning = static_binning(array, bins=_, **kwargs)
    else:
        raise ValueError(f"Binning {_} not understood.")
    return binning


def calculate_nd_bins(
    array: Optional[np.ndarray],
    bins=None,
    dim: Optional[int] = None,
    check_nan: bool = True,
    **kwargs,
) -> List[BinningBase]:
    """Find optimal binning from arguments (n-dimensional variant)

    Usage similar to `calculate_bins`.
    """
    if array is not None:
        if dim and array.shape[-1] != dim:
            raise ValueError(
                f"The array must be of shape (N, {dim}), {array.shape} found."
            )
        _, dim = array.shape

        if check_nan:
            if np.any(np.isnan(array)):
                raise ValueError("Cannot calculate bins in presence of NaN's.")

    # Prepare bins
    if isinstance(bins, list):
        if dim is not None:
            if len(bins) != dim:
                raise ValueError(
                    f"List of bins not understood, expected {dim} items, got {len(bins)}."
                )
        else:
            dim = len(bins)
    else:
        if not dim:
            raise ValueError(f"Required dim > 0: {dim}.")
        bins = [bins] * dim

    # Prepare arguments
    # TODO: Lists = argument for multiple axes, tuples = array argument
    range_ = kwargs.pop("range", None)
    if range_:
        if len(range_) == 2 and all(np.isscalar(i) for i in range_):
            range_ = dim * [range_]
        elif len(range_) != dim:
            raise ValueError("Wrong dimensionality of range")
    for key in list(kwargs.keys()):
        if isinstance(kwargs[key], list):
            if len(kwargs[key]) != dim:
                raise ValueError("Argument not understood.")
        else:
            kwargs[key] = dim * [kwargs[key]]

    if range_:
        kwargs["range"] = range_

    bins = [
        calculate_1d_bins(
            array[:, i] if array is not None else None,
            bins[i],
            **{k: kwarg[i] for k, kwarg in kwargs.items() if kwarg[i] is not None},
        )
        for i in range(dim)
    ]
    return bins
