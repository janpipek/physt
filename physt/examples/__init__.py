"""A set of examples used for demonstrating the physt capabilities / in tests."""
from __future__ import division
import numpy as np
from ..import h1, h2, h3


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
    data = np.random.normal(mean, sigma, (size,))
    return h1(data, name="normal", axis_name="x", title="1D normal distribution")


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
    data1 = np.random.normal(0, 1, (size,))
    data2 = np.random.normal(0, 1, (size,))
    return h2(data1, data2, name="normal", axis_names=tuple("xy"), title="2D normal distribution")


def normal_h3(size=10000):
    """A simple 3D histogram with normal distribution.

    Parameters
    ----------
    size : int
        Number of points

    Returns
    -------
    h : physt.histogram_nd.Histogram2D
    """
    data1 = np.random.normal(0, 1, (size,))
    data2 = np.random.normal(0, 1, (size,))
    data3 = np.random.normal(0, 1, (size,))
    return h3([data1, data2, data3], name="normal", axis_names=tuple("xyz"), title="3D normal distribution")
    
    
def fist():
    """A simple histogram in the shape of a fist.
    
    Returns
    -------
    h : physt.histogram1d.Histogram1D
    """
    import numpy as np
    from ..histogram1d import Histogram1D
    widths = [0, 1.2, 0.2, 1, 0.1, 1, 0.1, 0.9, 0.1, 0.8]
    edges = np.cumsum(widths)
    heights = np.asarray([4, 1, 7.5, 6, 7.6, 6, 7.5, 6, 7.2]) + 5
    return Histogram1D(edges, heights, axis_name="Is this a fist?", title="Physt \"logo\"")

ALL_EXAMPLES = [normal_h1, normal_h2, normal_h3, fist]


try:
    import pandas as pd

    def munros(edge_length=10):
        """Number of munros in different rectangular areas of Scotland.

        Parameters
        ----------
        edge_length : float
            Size of the rectangular grid in minutes.

        Returns
        -------
        h : physt.histogram_nd.Histogram2D
            Histogram in latitude and longitude.
        """
        data = load_dataset("munros")
        return h2(data["lat"], data["long"], "fixed_width", edge_length / 60, name="munros", title="Munros of Scotland")

    ALL_EXAMPLES.append(munros)

except ImportError:
    pass


def load_dataset(name):
    """Load example dataset.

    If seaborn is present, its datasets can be loaded.
    Physt also includes some datasets in CSV format.

    Parameters
    ----------
    name : str

    Returns
    -------
    dataset : pandas.DataFrame
    """
    # Our custom datasets:
    try:
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("Pandas not installed.")
        import pkgutil
        import io
        binary_data = pkgutil.get_data('physt', 'examples/{0}.csv'.format(name))
        return pd.read_csv(io.BytesIO(binary_data))
    except FileNotFoundError:
        pass

    # Seaborn datasets?
    try:
        import seaborn as sns
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name in sns.get_dataset_names():
                    return sns.load_dataset(name)
    except ImportError:
        pass

    # Fall through
    raise RuntimeError("Dataset {0} not available.".format(name))

try:
    import seaborn as sns

    def iris_h1(x="sepal_length"):
        """One-dimensional histogram of classical iris data.

        Parameters
        ----------
        x : str
            Name of the property to be histogrammed
            (sepal_length, sepal_width, petal_length, petal_width)
        """
        iris = load_dataset("iris")
        return h1(iris[x], "human", 20, name="iris")


    def iris_h2(x="sepal_length", y="sepal_width"):
        """Two-dimensional histogram of classical iris data.

        Parameters
        ----------
        x, y : str
            Names of the properties to be histogrammed
            (sepal_length, sepal_width, petal_length, petal_width)
        """
        iris = load_dataset("iris")
        return h2(iris[x], iris[y], "human", 20, name="iris")


    ALL_EXAMPLES += [iris_h1, iris_h2]

except ImportError:
    pass
