"""Functions to mimick histogram behaviour, only returning Histogram objects."""

import warnings

import numpy as np

from physt.histogram import Histogram
from physt.schema import NumpySchema, StaticSchema, Schema, MultiSchema

__all__ = ["histogram", "histogram2d", "histogramdd"]

BIN_COUNT_ALGORITHMS = ("auto", "fd", "doane", "scott",
                        "rice", "sturges", "sqrt")


def histogram(a, bins=10, range=None, normed=False, weights=None, density=None) -> Histogram:
    """Numpy-compatible 1D histogram.

    This functions supports the same parameters as numpy.histogram
    with the little exception of `normed` and `density` that have
    little meaning for the 

    Parameters
    ----------

    See Also
    --------
        np.histogram
        Histogram.normalize
        Histogram.densities
    """
    if normed:
        raise ValueError("The `normed` argument is known to be buggy in numpy and is not supported.")
    if density:
        import warnings
        warnings.warn("The `density` argument is ignored, please use .densities property of the histogram.")

    schema = make_numpy_schema(bins, range)
    values = schema.fit_and_apply(a, weights=weights)
    histogram = Histogram(schema, values)
    return histogram


def histogram2d(x, y, bins=10, range=None, normed=False, weights=None) -> Histogram:
    sample = np.asarray([x, y]).T
    return histogramdd(sample, bins, range=range, normed=normed, weights=weights)


def histogramdd(sample, bins=10, range=None, normed=False, weights=None) -> Histogram:
    if normed:
        raise ValueError("The `normed` argument is known to be buggy in numpy and is not supported.")
    from builtins import range as builtin_range
    sample = np.asarray(sample)
    ndim = sample.shape[1]
    if ndim > 10:
        raise ValueError("Histograms with dimension > 10 are not supported, {0} requsted.".format(ndim))
    if not range:
        range = [None] * ndim
    if isinstance(bins, (list, tuple)):
        if len(bins) != ndim:
            raise ValueError("List of bin arrays must contain {0} items, {1} found.".format(ndim, len(bins)))
        schemas = (make_numpy_schema(item, range=range[i], allow_string=False) for i, item in enumerate(bins))
    else:
        schemas = (make_numpy_schema(bins, range=range[i], allow_string=False) for i in builtin_range(ndim))

    schema = MultiSchema(schemas)
    values = schema.fit_and_apply(sample, weights=weights)
    histogram = Histogram(schema, values)
    return histogram


def make_numpy_schema(bins, range=None, allow_string:bool=True) -> Schema:
    """Create schema compatible with numpy function parameters.
    
    Parameters
    ----------
    allow_string
        If true, enable to specify the name of 
    
    See also
    --------
    numpy.histogram
    """
    if isinstance(bins, int):
        return NumpySchema(bins, range=range)
    if isinstance(bins, str):
        if allow_string:
            return NumpySchema(bins, range=range)
        else:
            raise ValueError("String not allowed as bin parameter")
    else:
        # TODO: Check properly
        bins = np.asarray(bins)
        return StaticSchema(edges=bins)
