"""
physt
=====

P(i/y)thon h(i/y)stograms. Inspired (and based on) numpy.histogram,
but designed for humans(TM) on steroids(TM).

(C) Jan Pipek, 2016-7, MIT licence
See https://github.com/janpipek/physt
"""

from __future__ import absolute_import
from . import binnings

__version__ = str('0.3.42')


def histogram(data, bins=None, *args, **kwargs):
    """Facade function to create 1D histograms.

    This proceeds in three steps:
    1) Based on magical parameter bins, construct bins for the histogram
    2) Calculate frequencies for the bins
    3) Construct the histogram object itself

    *Guiding principle:* parameters understood by numpy.histogram should be
    understood also by physt.histogram as well and should result in a Histogram1D
    object with (h.numpy_bins, h.frequencies) same as the numpy.histogram
    output. Additional functionality is a bonus.

    This function is also aliased as "h1".

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
    keep_missed: Optional[bool]
        store statistics about how many values were lower than limits
        and how many higher than limits (default: True)
    dropna: bool
        whether to clear data from nan's before histogramming
    name: str
        name of the histogram
    axis_name: str
        name of the variable on x axis
    adaptive: bool
        whether we want the bins to be modifiable
        (useful for continuous filling of a priori unknown data)
    dtype: type
        customize underlying data type: default int64 (without weight) or float (with weights)

    Other numpy.histogram parameters are excluded, see the methods of the Histogram1D class itself.

    Returns
    -------
    physt.histogram1d.Histogram1D

    See Also
    --------
    numpy.histogram
    """
    import numpy as np
    from .histogram1d import Histogram1D, calculate_frequencies
    from .binnings import calculate_bins

    adaptive = kwargs.pop("adaptive", False)
    dtype = kwargs.pop("dtype", None)

    if isinstance(data, tuple) and isinstance(data[0], str):    # Works for groupby DataSeries
        return histogram(data[1], bins, *args, name=data[0], **kwargs)
    elif type(data).__name__ == "DataFrame":
        raise RuntimeError("Cannot create histogram from a pandas DataFrame. Use Series.")

    # Collect arguments (not to send them to binning algorithms)
    dropna = kwargs.pop("dropna", True)
    weights = kwargs.pop("weights", None)
    keep_missed = kwargs.pop("keep_missed", True)
    name = kwargs.pop("name", None)
    axis_name = kwargs.pop("axis_name", None)

    # Convert to array
    if data is not None:
        array = np.asarray(data) #.flatten()
        if dropna:
            array = array[~np.isnan(array)]
    else:
        array = None

    # Get binning
    binning = calculate_bins(array, bins, *args,
                             check_nan=not dropna and array is not None,
                             adaptive=adaptive, **kwargs)
    # bins = binning.bins

    # Get frequencies
    if array is not None:
        (frequencies, errors2, underflow, overflow, stats) =\
            calculate_frequencies(array, binning=binning,
                                  weights=weights, dtype=dtype)
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
            axis_name = data.name
        elif hasattr(data, "fields") and len(data.fields) == 1 and isinstance(data.fields[0], str):
            # Case of dask fields (examples)
            axis_name = data.fields[0]
    return Histogram1D(binning=binning, frequencies=frequencies,
                       errors2=errors2, overflow=overflow,
                       underflow=underflow, stats=stats, dtype=dtype,
                       keep_missed=keep_missed, name=name, axis_name=axis_name)


def histogram2d(data1, data2, bins=10, *args, **kwargs):
    """Facade function to create 2D histograms.

    For implementation and parameters, see histogramdd.

    This function is also aliased as "h2".

    Returns
    -------
    physt.histogram_nd.Histogram2D

    See Also
    --------
    numpy.histogram2d
    histogramdd
    """
    import numpy as np

    # guess axis names
    if "axis_names" not in kwargs:
        if hasattr(data1, "name") and hasattr(data2, "name"):
            kwargs["axis_names"] = [data1.name, data2.name]
    if data1 is not None and data2 is not None:
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        data = np.concatenate([data1[:, np.newaxis],
                               data2[:, np.newaxis]], axis=1)
    else:
        data = None
    return histogramdd(data, bins, *args, dim=2, **kwargs)


def histogramdd(data, bins=10, *args, **kwargs):
    """Facade function to create n-dimensional histograms.

    3D variant of this function is also aliased as "h3".

    Parameters
    ----------
    data : array_like
        Container of all the values
    bins: Any
    weights: array_like, optional
        (as numpy.histogram)
    dropna: bool
        whether to clear data from nan's before histogramming
    name: str
        name of the histogram
    axis_names: Iterable[str]
        names of the variable on x axis
    adaptive:
        whether the bins should be updated when new non-fitting value are filled
    dtype: Optional[type]
        Underlying type for the histogram.
        If weights are specified, default is float. Otherwise int64
    dim: int
        Dimension - necessary if you are creating an empty adaptive histogram

    Returns
    -------
    physt.histogram_nd.HistogramND

    See Also
    --------
    numpy.histogramdd
    """
    import numpy as np
    from . import histogram_nd
    from .binnings import calculate_bins_nd

    adaptive = kwargs.pop("adaptive", False)
    dropna = kwargs.pop("dropna", True)
    name = kwargs.pop("name", None)
    dim = kwargs.pop("dim", None)
    axis_names = kwargs.pop("axis_names", None)

    # pandas - guess axis names
    if not "axis_names" in kwargs:
        if hasattr(data, "columns"):
            try:
                kwargs["axis_names"] = tuple(data.columns)
            except:
                pass # Perhaps columns has different meaning here.

    # Prepare and check data
    # Convert to array
    if data is not None:
        data = np.asarray(data)
        if data.ndim != 2:
            raise RuntimeError("Array must have shape (n, d)")
        if dim is not None and dim != data.shape[1]:
            raise RuntimeError("Dimension mismatch: {0}!={1}".format(dim, data.shape[1]))
        _, dim = data.shape
        if dropna:
            data = data[~np.isnan(data).any(axis=1)]
        check_nan = not dropna
    else:
        if dim is None:
            raise RuntimeError("You have to specify either data or its dimension.")
        data = np.zeros((0, dim))
        check_nan = False

    # Prepare bins
    bin_schemas = calculate_bins_nd(data, bins, *args, check_nan=check_nan, adaptive=adaptive,
                                    **kwargs)
    #bins = [binning.bins for binning in bin_schemas]

    # Prepare remaining data
    weights = kwargs.pop("weights", None)
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(data, ndim=dim,
                                                                      binnings=bin_schemas,
                                                                      weights=weights)

    kwargs["name"] = name
    if axis_names:
        kwargs["axis_names"] = axis_names
    if dim == 2:
        return histogram_nd.Histogram2D(binnings=bin_schemas, frequencies=frequencies,
                                        errors2=errors2, **kwargs)
    else:
        return histogram_nd.HistogramND(dimension=dim, binnings=bin_schemas,
                                        frequencies=frequencies, errors2=errors2, **kwargs)


# Aliases
h1 = histogram
h2 = histogram2d
h = histogramdd


def h3(data, *args, **kwargs):
    """Facade function to create 3D histograms.

    Parameters
    ----------
    data : array_like or list[array_like] or tuple[array_like]
        Can be a single array (with three columns) or three different arrays
        (for each component)

    Returns
    -------
    physt.histogram_nd.HistogramND
    """
    import numpy as np

    if data is not None and isinstance(data, (list, tuple)) and not np.isscalar(data[0]):
        if "axis_names" not in kwargs:
            kwargs["axis_names"] = [(column.name if hasattr(column, "name") else None) for column in data]
        data = np.concatenate([item[:, np.newaxis] for item in data], axis=1)
    else:
        kwargs["dim"] = 3    
    return histogramdd(data, *args, **kwargs)

from .special import polar_histogram
