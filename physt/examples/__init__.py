"""A set of examples used for demonstrating the physt capabilities / in tests."""


def normal_h1(size=10000, mean=0, sigma=1):
    """A simple 1D histogram with normal distribution.

    Parameters
    ----------
    size : int
        Number of points
    mean : float
        Mean of the distribution
    sigma : float
        Sigma of the distribution

    Returns
    -------
    h : physt.histogram1d.Histogram1D
    """
    import numpy as np
    from ..constructors import h1

    data = np.random.normal(mean, sigma, (size,))
    return h1(data)


def normal_h2(size=10000):
    """A simple 2D histogram with normal distribution.

    Parameters
    ----------
    size : int
        Number of points

    Returns
    -------
    h : physt.histogram_nd.Histogram2D
    """
    import numpy as np
    from ..constructors import h2

    data1 = np.random.normal(0, 1, (size,))
    data2 = np.random.normal(0, 1, (size,))
    return h2(data1, data2)

