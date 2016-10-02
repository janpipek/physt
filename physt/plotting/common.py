"""
Functions that are shared by several (all) plotting backends.

"""
from __future__ import absolute_import
import numpy as np


def get_data(h, density=False, cumulative=False, flatten=False):
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
            data = (h / h.total).cumulative_frequencies
        else:
            data = h.densities
    else:
        if cumulative:
            data = h.cumulative_frequencies
        else:
            data = h.frequencies

    if flatten:
        data = data.flatten()
    return data


def get_err_data(h, density=False, cumulative=False, flatten=False):
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
        data = h.errors / h.bin_sizes
    else:
        data = h.errors
    if flatten:
        data = data.flatten()
    return data
