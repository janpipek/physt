from typing import Optional, Iterable, Type, cast

import numpy as np

from physt.util import deprecation_alias
from physt.histogram1d import Histogram1D, calculate_frequencies
from physt.histogram_nd import HistogramND, Histogram2D
from physt.binnings import calculate_bins, calculate_bins_nd
from physt.histogram_collection import HistogramCollection
from physt.special_histograms import (
    polar,
    azimuthal,
    radial,
    cylindrical,
    cylindrical_surface,
    spherical,
    spherical_surface,
)
from physt.typing_aliases import ArrayLike, DtypeLike


def h1(
    data: Optional[ArrayLike],
    bins=None,
    *,
    adaptive: bool = False,
    dropna: bool = True,
    dtype: Optional[DtypeLike] = None,
    weights: Optional[ArrayLike] = None,
    keep_missed: bool = True,
    name: Optional[str] = None,
    title: Optional[str] = None,
    axis_name: Optional[str] = None,
    **kwargs
) -> Histogram1D:
    """Facade function to create 1D histograms.

    This proceeds in three steps:
    1) Based on magical parameter bins, construct bins for the histogram
    2) Calculate frequencies for the bins
    3) Construct the histogram object itself

    *Guiding principle:* parameters understood by numpy.histogram should be
    understood also by physt.histogram as well and should result in a Histogram1D
    object with (h.numpy_bins, h.frequencies) same as the numpy.histogram
    output. Additional functionality is a bonus.

    Parameters
    ----------
    data : array_like, optional
        Container of all the values (tuple, list, np.ndarray, pd.Series)
    bins: int or sequence of scalars or callable or str, optional
        If iterable => the bins themselves
        If int => number of bins for default binning
        If callable => use binning method (+ args, kwargs)
        If string => use named binning method (+ args, kwargs)
    weights: array_like, optional
        (as numpy.histogram)
    keep_missed: Store statistics about how many values were lower than limits
        and how many higher than limits (default: True)
    dropna: Whether to clear data from nan's before histogramming
    name: Name of the histogram
    title: What will be displayed in the title of the plot
    axis_name: Name of the variable on x axis
    adaptive: Whether we want the bins to be modifiable
        (useful for continuous filling of a priori unknown data)
    dtype: Customize underlying data type: default int64 (without weight) or float (with weights)

    Other numpy.histogram parameters are excluded, see the methods of the Histogram1D class itself.

    See Also
    --------
    numpy.histogram
    """

    if isinstance(data, tuple) and isinstance(data[0], str):  # Works for groupby DataSeries
        return h1(data[1], bins, name=data[0], **kwargs)
    if type(data).__name__ == "DataFrame":
        raise TypeError("Cannot create histogram from a pandas DataFrame. Use Series.")

    # Convert to array
    if data is not None:
        array = np.asarray(data)  # .flatten()
        if dropna:
            array = array[~np.isnan(array)]
    else:
        array = None

    # Get binning
    binning = calculate_bins(
        array, bins, check_nan=not dropna and array is not None, adaptive=adaptive, **kwargs
    )
    # bins = binning.bins

    # Get frequencies
    if array is not None:
        (frequencies, errors2, underflow, overflow, stats) = calculate_frequencies(
            array, binning=binning, weights=weights, dtype=dtype
        )
    else:
        frequencies = None
        errors2 = None
        underflow = 0
        overflow = 0
        stats = {"sum": 0.0, "sum2": 0.0}

    # Construct the object
    if not keep_missed:
        underflow = 0
        overflow = 0
    if not axis_name:
        if hasattr(data, "name"):
            axis_name = str(data.name)  # type: ignore
        elif (
            hasattr(data, "fields")
            and len(data.fields) == 1  # type: ignore
            and isinstance(data.fields[0], str)  # type: ignore
        ):
            # Case of dask fields (examples)
            axis_name = str(data.fields[0])  # type: ignore
    return Histogram1D(
        binning=binning,
        frequencies=frequencies,
        errors2=errors2,
        overflow=overflow,
        underflow=underflow,
        stats=stats,
        dtype=dtype,
        keep_missed=keep_missed,
        name=name,
        axis_name=axis_name,
        title=title,
    )


def h2(data1: Optional[ArrayLike], data2: Optional[ArrayLike], bins=10, **kwargs) -> Histogram2D:
    """Facade function to create 2D histograms.

    For implementation and parameters, see histogramdd.

    See Also
    --------
    numpy.histogram2d
    histogramdd
    """
    # guess axis names
    if "axis_names" not in kwargs:
        if hasattr(data1, "name") and hasattr(data2, "name"):
            kwargs["axis_names"] = [str(data1.name), str(data2.name)]  # type: ignore
    if data1 is not None and data2 is not None:
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        data = np.concatenate([data1[:, np.newaxis], data2[:, np.newaxis]], axis=1)
    else:
        data = None
    return cast(Histogram2D, h(data, bins, dim=2, **kwargs))


def h3(data: Optional[ArrayLike], bins=None, **kwargs) -> HistogramND:
    """Facade function to create 3D histograms.

    Parameters
    ----------
    data : array_like or list[array_like] or tuple[array_like]
        Can be a single array (with three columns) or three different arrays
        (for each component)
    """
    if data is not None and isinstance(data, (list, tuple)) and not np.isscalar(data[0]):
        if "axis_names" not in kwargs:
            kwargs["axis_names"] = [
                (column.name if hasattr(column, "name") else None) for column in data
            ]
        data = np.concatenate([item[:, np.newaxis] for item in data], axis=1)
    else:
        kwargs["dim"] = 3
    return h(data, bins, **kwargs)


def h(
    data: Optional[ArrayLike],
    bins=10,
    *,
    adaptive=False,
    dropna: bool = True,
    name: Optional[str] = None,
    title: Optional[str] = None,
    axis_names: Optional[Iterable[str]] = None,
    dim: Optional[int] = None,
    weights: Optional[ArrayLike] = None,
    **kwargs
) -> HistogramND:
    """Facade function to create n-dimensional histograms.

    3D variant of this function is also aliased as "h3".

    Parameters
    ----------
    data : array_like
        Container of all the values
    bins: Any
    weights: array_like, optional
        (as numpy.histogram)
    dropna: Whether to clear data from nan's before histogramming
    name: Name of the histogram
    axis_names: Names of the variable on x axis
    adaptive: Whether the bins should be updated when new non-fitting value are filled
    dtype: Optional[type]
        Underlying type for the histogram.
        If weights are specified, default is float. Otherwise int64
    title: What will be displayed in the title of the plot
    dim: Dimension - necessary if you are creating an empty adaptive histogram

    Note: For most arguments, if a list is passed, its values are used as values for
    individual axes.

    See Also
    --------
    numpy.histogramdd
    """
    # pandas - guess axis names
    if not axis_names:
        if hasattr(data, "columns"):
            try:
                axis_names = tuple(str(c) for c in data.columns)  # type: ignore
            except:
                pass  # Perhaps columns has different meaning here.

    # Prepare and check data
    # Convert to array
    if data is not None:
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("Array must have shape (n, d), {0} encountered".format(data.shape))
        if dim is not None and dim != data.shape[1]:
            raise ValueError("Dimension mismatch: {0}!={1}".format(dim, data.shape[1]))
        _, dim = data.shape
        if dropna:
            data = data[~np.isnan(data).any(axis=1)]
        check_nan = not dropna
    else:
        if dim is None:
            raise ValueError("You have to specify either data or its dimension.")
        check_nan = False

    # Prepare bins
    bin_schemas = calculate_bins_nd(
        data, bins, dim=dim, check_nan=check_nan, adaptive=adaptive, **kwargs
    )

    # Prepare remaining data
    klass: Type[HistogramND] = Histogram2D if dim == 2 else HistogramND  # type: ignore
    if name:
        kwargs["name"] = name
    if title:
        kwargs["title"] = title
    return klass.from_calculate_frequencies(
        data, binnings=bin_schemas, weights=weights, axis_names=axis_names, **kwargs
    )


# Aliases
histogram = deprecation_alias(h1, "histogram")
histogram2d = deprecation_alias(h2, "histogram2d")
histogramdd = deprecation_alias(h, "histogramdd")


def collection(data, bins=10, **kwargs) -> HistogramCollection:
    """Create histogram collection with shared binnning."""
    if hasattr(data, "columns"):
        data = {column: data[column] for column in data.columns}
    return HistogramCollection.multi_h1(data, bins, **kwargs)


__all__ = [
    "h1",
    "h2",
    "h3",
    "histogram",
    "histogram2d",
    "histogramdd",
    "collection",
    "polar",
    "azimuthal",
    "radial",
    "cylindrical",
    "cylindrical_surface",
    "spherical",
    "spherical_surface",
]
