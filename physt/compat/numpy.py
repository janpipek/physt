"""Functions to mimick histogram behaviour, only returning Histogram objects."""

import warnings

import numpy as np

from physt.histogram import Histogram
from physt.schema import NumpySchema, StaticSchema, Schema

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
    if isinstance(bins, np.ndarray):
        pass
        
    elif isinstance(bins, (list, tuple)):
        pass 

def histogramdd(sample, bins=10, range=None, normed=False, weights=None) -> Histogram:
    pass
    

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
            raise ValueError()
    else:
        # TODO: Check properly
        bins = np.asarray(bins)
        return StaticSchema(edges=bins)
