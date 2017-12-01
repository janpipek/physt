"""Functions to mimick histogram behaviour, only returning Histogram objects."""

from physt.histogram import Histogram
from physt.builder import HistogramBuilder
from physt.schema import NumpySchema, StaticSchema

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
    if isinstance(bins, (str, int)):
        schema = NumpySchema(bins, range=range)
    else:
        schema = StaticSchema(edges=bins)
    builder = HistogramBuilder((schema,))
    h = builder(a, weights=weights)
    if normed:
        raise ValueError("The `normed` argument is known to be buggy in numpy and is not supported.")
    if density:
        import warnings
        warnings.warn("The `density` argument is ignored, please use .densities property of the histogram.")
    return h

def histogram2d():
    pass

def histogramdd():
    pass
