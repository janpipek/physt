"""Methods for investigation and manipulation of bin arrays."""
from __future__ import absolute_import
import numpy as np


def make_bin_array(bins):
    """Turn bin data into array understood by HistogramXX classes.

    Parameters
    ----------
    bins: array_like
        Array of edges or array of edge tuples

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
        # if bins.shape[0] == 0:
        #     raise RuntimeError("Needs at least one bin")
        return np.hstack((bins[:-1, np.newaxis], bins[1:, np.newaxis]))
    elif bins.ndim == 2:
        if bins.shape[1] != 2:
            raise RuntimeError("Binning schema with ndim==2 must have 2 columns")
        # if bins.shape[0] == 0:
        #     raise RuntimeError("Needs at least one bin")
        return bins  # Already correct, just pass
    else:
        raise RuntimeError("Binning schema must have ndim==1 or ndim==2")


def to_numpy_bins(bins):
    """Convert physt bin format to numpy edges.

    Parameters
    ----------
    bins: array_like
        1-D (n) or 2-D (n, 2) array of edges

    Returns
    -------
    edges: np.ndarray
        all edges
    """
    bins = np.asarray(bins)
    if bins.ndim == 1:     # Already in the proper format
        return bins
    if not is_consecutive(bins):
        raise RuntimeError("Cannot create numpy bins from inconsecutive edges")
    return np.concatenate([bins[:1, 0], bins[:, 1]])


def to_numpy_bins_with_mask(bins):
    """Numpy binning edges including gaps.

    Parameters
    ----------
    bins: array_like
        1-D (n) or 2-D (n, 2) array of edges

    Returns
    -------
    edges: np.ndarray
        all edges
    mask: np.ndarray
        List of indices that correspond to bins that have to be included

    Examples
    --------
    >>> to_numpy_bins_with_mask([0, 1, 2])
    (array([0.,   1.,   2.]), array([0, 1]))

    >>> to_numpy_bins_with_mask([[0, 1], [2, 3]])
    (array([0, 1, 2, 3]), array([0, 2])
    """
    bins = np.asarray(bins)
    if bins.ndim == 1:
        edges = bins
        if bins.shape[0] > 1:
            mask = np.arange(bins.shape[0] - 1)
        else:
            mask = []
    elif bins.ndim == 2:
        edges = []
        mask = []
        j = 0
        if bins.shape[0] > 0:
            edges.append(bins[0, 0])
            for i in range(bins.shape[0] - 1):
                mask.append(j)
                edges.append(bins[i, 1])
                if bins[i, 1] != bins[i+1, 0]:
                    edges.append(bins[i+1, 0])
                    j += 1
                j += 1
            mask.append(j)
            edges.append(bins[-1, 1])
    else:
        raise RuntimeError("to_numpy_bins_with_mask: array with dim=1 or 2 expected")
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError("to_numpy_bins_with_mask: edges array not monotone.")
    return edges, mask


def is_rising(bins):
    """Check whether the bins are in raising order.

    Does not check if the bins are consecutive.

    Parameters
    ----------
    bins: array_like

    Returns
    -------
    bool
    """
    # TODO: Optimize for numpy bins
    bins = make_bin_array(bins)
    if np.any(bins[:, 0] >= bins[:, 1]):
        return False
    if np.any(bins[1:, 0] < bins[:-1, 1]):
        return False
    return True


def is_consecutive(bins, rtol=1.e-5, atol=1.e-8):
    """Check whether the bins are consecutive (edges match).

    Does not check if the bins are in rising order.

    Returns
    -------
    bool
    """
    bins = np.asarray(bins)
    if bins.ndim == 1:
        return True
    else:
        bins = make_bin_array(bins)
        return np.allclose(bins[1:, 0], bins[:-1, 1], rtol, atol)


def is_bin_subset(sub, sup):
    """Check whether all bins in one binning are present also in another:

    Parameters
    ----------
    sub: array_like
        Candidate for the bin subset
    sup: array_like
        Candidate for the bin superset

    Returns
    -------
    bool

    """
    sub = make_bin_array(sub)
    sup = make_bin_array(sup)

    for row in sub:
        if not (row == sup).all(axis=1).any():
            # TODO: Enable also approximate equality
            return False
    return True


def is_bin_superset(sup, sub):
    """Inverse of is_bin_subset"""
    return is_bin_subset(sub=sub, sup=sup)
