__version__ = str('0.1.0')
from .histogram1d import Histogram1D
import numpy as np


def histogram(data, bins=50, **kwargs):
    """Create a histogram from data.

    Parameters
    ----------
    data : array_like
        (as numpy.histogram)
    bins: int or sequence of scalars, optional
        (as numpy.histogram)
    range: tuple, optional
        (as numpy.histogram) - first left edge, last right edge
    weights: array_like, optional
        (as numpy.histogram)
    keep_missed: bool, optional
        store statistics about how many values were lower than limits and how many higher than limits (default: True)

    Other numpy.histogram parameters are excluded, see the methods of the Histogram1D class itself.

    Returns
    -------
    Histogram1D
    """
    if isinstance(data, np.ndarray):
        np_kwargs = {key : kwargs.get(key) for key in kwargs if key in kwargs and key in ["range", "weights"]}
        np_values, np_bins = np.histogram(data, bins, **np_kwargs)
        h_kwargs = {}
        if kwargs.get("keep_missed", True):
            weights = kwargs.get("weights", np.ones(data.shape))
            h_kwargs["underflow"] = weights[data < np_bins[0]].sum()
            h_kwargs["overflow"] = weights[data > np_bins[-1]].sum()
        return Histogram1D(np_bins, np_values, **h_kwargs)
    else:
        return histogram(np.array(data), bins, **kwargs)