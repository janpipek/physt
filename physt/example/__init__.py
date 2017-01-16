"""A set of examples used for demonstrating the physt capabilities / in tests."""

import numpy as np
from ..import h1, h2


def normal_h1():
    """A simple 1D histogram with normal distribution."""
    data = np.random.normal(0, 1, (10000,))
    return h1(data)

def normal_h2():
    """A simple 2D histogram with normal distribution."""
    data1 = np.random.normal(0, 1, (10000,))
    data2 = np.random.normal(0, 1, (10000,))
    return h2(data1, data2)
