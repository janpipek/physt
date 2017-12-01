"""Functions to mimick histogram behaviour, only returning Histogram objects."""

from physt.histogram import Histogram
from physt.builder import HistogramBuilder
from physt.schema import NumpySchema, StaticSchema, Schema

__all__ = ["histogram", "histogram2d", "histogramdd"]


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
    schema = _make_numpy_schema(bins, range)
    builder = HistogramBuilder((schema,))
    h = builder.apply(a, weights=weights)
    if normed:
        raise ValueError("The `normed` argument is known to be buggy in numpy and is not supported.")
    if density:
        import warnings
        warnings.warn("The `density` argument is ignored, please use .densities property of the histogram.")
    return h

def _make_numpy_schema(bins, range=None, allow_string=True) -> Schema:
    if isinstance(bins, (int)) or (allow_string and isinstance(bins, str)):
        return NumpySchema(bins, range=range)
    else:
        # TODO: Check properly
        bins = np.asarray(bins)
        return StaticSchema(edges=bins)    

def histogram2d(histogram2d(x, y, bins=10, range=None, normed=False, weights=None):
    if isinstance(bins, np.ndarray):
        
    elif isinstance(bins, (list, tuple)):


def histogramdd(sample, bins=10, range=None, normed=False, weights=None):
    pass
    
