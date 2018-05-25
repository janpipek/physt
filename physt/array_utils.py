import numpy as np


class NonConsecutiveBinsError(RuntimeError):
    pass


def make_2d_array(ndim:int, *arrays) -> np.ndarray:
    """Take one 2D array or multiple 1D arrays and turn them into
    2D array of entries to multidimensional histogram.


    """
    if ndim < 2:
        raise ValueError("make_2d_array")
    if len(arrays) == 1:
        array = np.asarray(arrays[0])
    elif len(arrays) == ndim:
        numpy_arrays = [np.asarray(a) for a in arrays]
        array = np.asarray(numpy_arrays).T # TODO: More clever concat!
    else:
        raise ValueError("make_2d_array({0}, ...) requires 1 or {0} arguments, {1} supplied.".format(ndim, len(arrays)))

    if array.ndim != 2 or array.shape[1] != ndim:
        raise ValueError("At least one of the arrays supplied was invalid.")
    return array


def make_bin_array(bins):
    """Turn edges or bins to just bins.

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
            raise RuntimeError(
                "Bin array with ndim==2 must have 2 columns")
        # if bins.shape[0] == 0:
        #     raise RuntimeError("Needs at least one bin")
        return bins  # Already correct, just pass
    else:
        raise RuntimeError("Bin array must have ndim==1 or ndim==2")


def bins_to_edges(bins) -> np.ndarray:
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
    if bins.ndim == 1:  # Already in the proper format
        return bins
    if not is_consecutive(bins):
        raise NonConsecutiveBinsError(
            "Cannot create numpy bins from inconsecutive edges")
    return np.concatenate([bins[:1, 0], bins[:, 1]])


def bins_to_edges_and_mask(bins) -> np.ndarray:
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
                if bins[i, 1] != bins[i + 1, 0]:
                    edges.append(bins[i + 1, 0])
                    j += 1
                j += 1
            mask.append(j)
            edges.append(bins[-1, 1])
    else:
        raise RuntimeError(
            "to_numpy_bins_with_mask: array with dim=1 or 2 expected")
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(
            "to_numpy_bins_with_mask: edges array not monotone.")
    return edges, mask


def is_rising(bins) -> bool:
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


def is_consecutive(bins, rtol:float=1.e-5, atol:float=1.e-8):
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