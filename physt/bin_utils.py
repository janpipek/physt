import numpy as np


def make_bin_array(bins):
    """Turn bin data into array understood by HistogramXX classes.

    Parameters
    ----------
    bins: array_like
        Array of edges or array of edge tuples.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> make_bin_array([0, 1, 2])
    array([[0, 1],
           [1, 2]])
    >>> make_bin_array([[0, 1], [2, 3]])
    array([[0, 1],
           [2, 3]])
    """
    bins = np.asarray(bins)
    if bins.ndim == 1:
        return np.hstack((bins[:-1,np.newaxis], bins[1:,np.newaxis]))
    elif bins.ndim == 2:
        if bins.shape[1] != 2:
            raise RuntimeError("Binning schema with ndim==2 must have 2 columns")
        return bins  # Already correct, just pass
    else:
        raise RuntimeError("Binning schema must have ndim==1 or ndim==2")


def is_rising(bins):
    """Check whether the bins are in raising order.

    Does not check if the bins are consecutive.

    Returns
    -------
    bool
    """
    bins = make_bin_array(bins)
    if np.any(bins[:,0] >= bins[:,1]):
        return False
    if np.any(bins[1:,0] < bins[:-1,1]):
        return False
    return True


def is_consecutive(bins, rtol=1.e-5, atol=1.e-8):
    """Check whether the bins are consecutive (edges match).

    Does not check if the bins are in rising order.

    Returns
    -------
    bool
    """
    if bins.ndim == 1:
        return True
    else:
        bins = make_bin_array(bins)
        return np.allclose(bins[1:,0], bins[:-1,1], rtol, atol)