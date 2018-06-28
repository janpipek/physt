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
        Whether to return cumulative sums instead of individual
    flatten : bool
        Whether to flatten multidimensional bins

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
        Whether to return cumulative sums instead of individual
    flatten : bool
        Whether to flatten multidimensional bins

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


def get_value_format(value_format=str):
    """Create a formatting function from a generic value_format argument.
    
    Parameters
    ----------
    value_format : str or Callable

    Returns
    -------
    Callable
    """
    if value_format is None:
        value_format = ""
    if isinstance(value_format, str):
        format_str = "{0:" + value_format + "}"
        value_format = lambda x: format_str.format(x)
    
    return value_format


def pop_kwargs_with_prefix(prefix, kwargs):
    """Pop all items from a dictionary that have keys beginning with a prefix.

    Parameters
    ----------
    prefix : str
    kwargs : dict

    Returns
    -------
    kwargs : dict
        Items popped from the original directory, with prefix removed.
    """
    keys = [key for key in kwargs if key.startswith(prefix)]
    return {key[len(prefix):]: kwargs.pop(key) for key in keys}