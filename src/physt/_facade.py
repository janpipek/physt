"""Facade functions that allow to compute and create histograms without explicit instance creation.

This involves:
- finding proper bins
- calculating frequencies
- creating the proper histogram instances

Note that the histogram classes are rather data structures and need computed data to be created.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from physt._construction import (
    calculate_1d_bins,
    calculate_1d_frequencies,
    calculate_nd_bins,
    extract_1d_array,
    extract_and_concat_arrays,
    extract_axis_name,
    extract_axis_names,
    extract_nd_array,
    extract_weights,
)
from physt._util import deprecation_alias
from physt.special_histograms import (
    azimuthal,
    cylindrical,
    cylindrical_surface,
    polar,
    radial,
    spherical,
    spherical_surface,
)
from physt.types import Histogram1D, Histogram2D, HistogramCollection, HistogramND

if TYPE_CHECKING:
    from typing import Iterable, Optional, Type

    from physt.typing_aliases import ArrayLike, DTypeLike


def h1(
    data: Optional[ArrayLike],
    bins=None,
    *,
    adaptive: bool = False,
    dropna: bool = True,
    dtype: Optional[DTypeLike] = None,
    weights: Optional[ArrayLike] = None,
    keep_missed: bool = True,
    name: Optional[str] = None,
    title: Optional[str] = None,
    axis_name: Optional[str] = None,
    **kwargs,
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

    # Extra treatment for pandas types
    if isinstance(data, tuple) and isinstance(
        data[0], str
    ):  # Works for groupby DataSeries
        return h1(data[1], bins, name=data[0], **kwargs)
    if type(data).__name__ == "DataFrame":
        raise TypeError(
            "Cannot create a 1D histogram from a pandas DataFrame. Use Series."
        )

    array, array_mask = extract_1d_array(data, dropna=dropna)

    weights = extract_weights(weights, array_mask=array_mask)

    binning = calculate_1d_bins(
        array,
        bins,
        check_nan=not dropna and array is not None,
        adaptive=adaptive,
        **kwargs,
    )

    axis_name = extract_axis_name(data, axis_name=axis_name)

    frequencies, errors2, underflow, overflow, stats = calculate_1d_frequencies(
        data=array,
        binning=binning,
        weights=weights,
        dtype=dtype,
    )

    return Histogram1D(
        binning=binning,
        frequencies=frequencies,
        errors2=errors2,
        underflow=underflow,
        overflow=overflow,
        dtype=dtype,
        stats=stats,
        keep_missed=keep_missed,
        name=name,
        axis_name=axis_name,
        title=title,
    )


def h2(
    data1: Optional[ArrayLike], data2: Optional[ArrayLike], bins=10, **kwargs
) -> Histogram2D:
    """Facade function to create 2D histograms.

    For implementation and parameters, see histogramdd.

    See Also
    --------
    numpy.histogram2d
    histogramdd
    """
    # guess axis names
    if "axis_names" not in kwargs:
        kwargs["axis_names"] = tuple(extract_axis_name(data) for data in (data1, data2))
    data, _ = extract_and_concat_arrays(data1, data2, dropna=False)
    result = h(data, bins, dim=2, **kwargs)
    return cast(Histogram2D, result)


def h3(data: Optional[ArrayLike], bins=None, **kwargs) -> HistogramND:
    """Facade function to create 3D histograms.

    Parameters
    ----------
    data : array_like or list[array_like] or tuple[array_like]
        Can be a single array (with three columns) or three different arrays
        (for each component)
    """
    if (
        data is not None
        and isinstance(data, (list, tuple))
        and not np.isscalar(data[0])
    ):
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
    adaptive: bool = False,
    dropna: bool = True,
    name: Optional[str] = None,
    title: Optional[str] = None,
    axis_names: Optional[Iterable[str]] = None,
    dim: Optional[int] = None,
    weights: Optional[ArrayLike] = None,
    **kwargs,
) -> HistogramND:
    """Facade function to create n-dimensional histograms.

    3D variant of this function is also aliased as "h3".

    Parameters
    ----------
    data: Container of all the values
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
    axis_names = extract_axis_names(data, axis_names=axis_names)
    check_nan = data is not None and not dropna

    dim, array, array_mask = extract_nd_array(data, dim=dim, dropna=dropna)

    weights = extract_weights(weights, array_mask=array_mask)

    bin_schemas = calculate_nd_bins(
        array, bins, dim=dim, check_nan=check_nan, adaptive=adaptive, **kwargs
    )

    # Prepare remaining data
    klass: Type[HistogramND] = Histogram2D if dim == 2 else HistogramND  # type: ignore
    if name:
        kwargs["name"] = name
    if title:
        kwargs["title"] = title
    return klass.from_calculate_frequencies(
        array,
        binnings=bin_schemas,
        weights=weights,
        axis_names=axis_names,
        name=name,
        title=title,
    )


# Aliases
histogram = h1
histogram2d = deprecation_alias(h2, "histogram2d")
histogramdd = deprecation_alias(h, "histogramdd")


def collection(data, bins=10, **kwargs) -> HistogramCollection:
    """Create histogram collection with shared binning."""
    if hasattr(data, "columns"):
        data = {column: data[column] for column in data.columns}
    return HistogramCollection.multi_h1(data, bins, **kwargs)


__all__ = [
    "azimuthal",
    "collection",
    "cylindrical_surface",
    "cylindrical",
    "h",
    "h1",
    "h2",
    "h3",
    "histogram",
    "histogram2d",
    "histogramdd",
    "polar",
    "radial",
    "spherical_surface",
    "spherical",
]
