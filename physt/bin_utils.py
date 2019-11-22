"""Methods for investigation and manipulation of bin arrays."""
from typing import Optional, Tuple

import numpy as np


def make_bin_array(bins) -> np.ndarray:
    """Turn bin data into array understood by HistogramXX classes.

    Parameters
    ----------
    bins: array_like
        Array of edges or array of edge tuples

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


def to_numpy_bins(bins) -> np.ndarray:
    """Convert physt bin format to numpy edges.

    Parameters
    ----------
    bins: array_like
        1-D (n) or 2-D (n, 2) array of edges

    Returns
    -------
    edges: all edges
    """
    bins = np.asarray(bins)
    if bins.ndim == 1:     # Already in the proper format
        return bins
    if not is_consecutive(bins):
        raise RuntimeError("Cannot create numpy bins from inconsecutive edges")
    return np.concatenate([bins[:1, 0], bins[:, 1]])


def to_numpy_bins_with_mask(bins) -> Tuple[np.ndarray, np.ndarray]:
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


def is_rising(bins) -> bool:
    """Check whether the bins are in raising order.

    Does not check if the bins are consecutive.

    Parameters
    ----------
    bins: array_like
    """
    # TODO: Optimize for numpy bins
    bins = make_bin_array(bins)
    if np.any(bins[:, 0] >= bins[:, 1]):
        return False
    if np.any(bins[1:, 0] < bins[:-1, 1]):
        return False
    return True


def is_consecutive(bins, rtol: float = 1.e-5, atol: float = 1.e-8) -> bool:
    """Check whether the bins are consecutive (edges match).

    Does not check if the bins are in rising order.
    """
    bins = np.asarray(bins)
    if bins.ndim == 1:
        return True
    else:
        bins = make_bin_array(bins)
        return np.allclose(bins[1:, 0], bins[:-1, 1], rtol, atol)


def is_bin_subset(sub, sup) -> bool:
    """Check whether all bins in one binning are present also in another:

    Parameters
    ----------
    sub: array_like
        Candidate for the bin subset
    sup: array_like
        Candidate for the bin superset
    """
    sub = make_bin_array(sub)
    sup = make_bin_array(sup)

    for row in sub:
        if not (row == sup).all(axis=1).any():
            # TODO: Enable also approximate equality
            return False
    return True


def is_bin_superset(sup, sub) -> bool:
    """Inverse of is_bin_subset"""
    return is_bin_subset(sub=sub, sup=sup)


def find_human_width_decimal(raw_width: float) -> float:
    subscales = np.array([0.5, 1, 2, 2.5, 5, 10])
    power = np.floor(np.log10(raw_width)).astype(int)
    best_index = np.argmin(np.abs(np.log(subscales * (10.0 ** power) / raw_width)))
    return (10.0 ** power) * subscales[best_index]


def find_human_width_60(raw_width: float) -> int:
    subscales = np.array([1, 2, 5, 10, 15, 20, 30])
    best_index = np.argmin(np.abs(np.log(subscales / raw_width)))
    return subscales[best_index]


def find_human_width_24(raw_width: float) -> int:
    subscales = np.array([1, 2, 3, 4, 6, 12])
    best_index = np.argmin(np.abs(np.log(subscales / raw_width)))
    return subscales[best_index]


def find_human_width(raw_width: float, kind: Optional[str] = None) -> float:
    if not kind:
        return find_human_width_decimal(raw_width)
    elif kind == "time":
        if raw_width < 0.8:
            return find_human_width_decimal(raw_width)
        elif raw_width < 50:
            return find_human_width_60(raw_width)
        elif raw_width < 3000:
            return find_human_width_60(raw_width / 60) * 60
        elif raw_width < 70000:
            return find_human_width_24(raw_width / 3600) * 3600
        else:
            return find_human_width_decimal(raw_width / 86400) * 86400
    else:
        raise ValueError("Value of 'kind' not understood: '{0}'.".format(kind))