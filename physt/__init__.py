from . import binning, histogram1d

import numpy as np

__version__ = str('0.1.0')


def histogram(data, bins=None, **kwargs):
    """Create a histogram from data.

    Parameters
    ----------
    data : array_like
        (as numpy.histogram)
    bins: int or sequence of scalars or callable, optional
        (as numpy.histogram)
    range: tuple, optional
        (as numpy.histogram) - first left edge, last right edge
    weights: array_like, optional
        (as numpy.histogram)
    method: str or callable
        Binning method
    keep_missed: bool, optional
        store statistics about how many values were lower than limits and how many higher than limits (default: True)

    Other numpy.histogram parameters are excluded, see the methods of the Histogram1D class itself.

    Returns
    -------
    Histogram1D

    See Also
    --------
    numpy.histogram
    """
    if isinstance(data, np.ndarray):
        if bins is None:
            nbins = binning.ideal_bin_count(data=data)
            bins = binning.numpy_like(data, nbins, **kwargs)
        elif np.iterable(bins):
            bins = np.array(bins)
        elif isinstance(bins, int):
            bins = binning.numpy_like(data, bins, **kwargs)
        elif callable(bins):
            bins = bins(data, **kwargs)
        constructor_args = histogram1d.get_histogram_data(data,
                                              bins=bins,
                                              weights=kwargs.get("weights", None),
                                              keep_missed=kwargs.get("keep_missed", True))
        return histogram1d.Histogram1D(**constructor_args)
    else:
        return histogram(np.array(data), bins, **kwargs)