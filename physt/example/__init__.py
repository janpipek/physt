"""A set of examples used for demonstrating the physt capabilities / in tests."""

import numpy as np
from ..import h1, h2


def normal_h1(size=10000):
    """A simple 1D histogram with normal distribution."""
    data = np.random.normal(0, 1, (size,))
    return h1(data)

def normal_h2(size=10000):
    """A simple 2D histogram with normal distribution."""
    data1 = np.random.normal(0, 1, (size,))
    data2 = np.random.normal(0, 1, (size,))
    return h2(data1, data2)

def iris_h1(x="sepal_length"):
    """1D histogram of iris data."""
    try:
        import seaborn as sns
    except ImportError:
        raise RuntimeError("Cannot plot iris data, seaborn has to be present.")
    iris = sns.load_dataset("iris")
    return h1(iris[x], "human", 20, name="iris")


def iris_h2(x="sepal_length", y="sepal_width"):
    """1D histogram of iris data."""
    try:
        import seaborn as sns
    except ImportError:
        raise RuntimeError("Cannot plot iris data, seaborn has to be present.")
    iris = sns.load_dataset("iris")
    return h2(iris[x], iris[y], "human", 20, name="iris")


ALL_EXAMPLES = (normal_h1, normal_h2, iris_h1, iris_h2)
