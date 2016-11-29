"""
Functions that are shared by several (all) plotting backends.

"""
from __future__ import absolute_import


def get_data(histogram, density=False, cumulative=False, flatten=False):
    """Get histogram data based on plotting parameters.

    Parameters
    ----------
    h : physt.histogram_base.HistogramBase
    density : bool
        Whether to divide bin contents by bin size
    cumulative : bool
        Where to return cumulative sums instead of individual
    flatten : bool
        Where to flatten multidimensional bins

    Returns
    -------
    np.ndarray

    """
    if density:
        if cumulative:
            data = (histogram / histogram.total).cumulative_frequencies
        else:
            data = histogram.densities
    else:
        if cumulative:
            data = histogram.cumulative_frequencies
        else:
            data = histogram.frequencies

    if flatten:
        data = data.flatten()
    return data


def get_err_data(histogram, density=False, cumulative=False, flatten=False):
    """Get histogram error data based on plotting parameters.

    Parameters
    ----------
    h : physt.histogram_base.HistogramBase
    density : bool
        Whether to divide bin contents by bin size
    cumulative : bool
        Where to return cumulative sums instead of individual
    flatten : bool
        Where to flatten multidimensional bins

    Returns
    -------
    np.ndarray
    """
    if cumulative:
        raise RuntimeError("Error bars not supported for cumulative plots.")
    if density:
        data = histogram.errors / histogram.bin_sizes
    else:
        data = histogram.errors
    if flatten:
        data = data.flatten()
    return data
