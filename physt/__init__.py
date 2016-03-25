__version__ = str('0.1.0')
from .histogram1d import Histogram1D
import numpy as np


def histogram(data, bins=50, **kwargs):
    """Create a histogram from data.

    :param data: A compatible object (numpy array-like)
    """
    if isinstance(data, np.ndarray):
        np_values, np_bins = np.histogram(data, bins, **kwargs)
        return Histogram1D(np_bins, np_values)
    # elseif pandas, ...
    else:
        return histogram(np.array(data), bins, **kwargs)