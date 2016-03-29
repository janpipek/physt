import numpy as np


class Histogram1D(object):
    """Representation of one-dimensional histogram.
    """
    def __init__(self, bins, frequencies=None):
        bins = np.array(bins)
        if bins.ndim == 1:       # Numpy-style
            self._bins = np.hstack((bins[:-1,np.newaxis], bins[1:,np.newaxis]))
        elif bins.ndim == 2:     # Tuple-style
            if bins.shape[1] != 2:
                raise RuntimeError("")
            self._bins = bins
        else:
            raise RuntimeError("Unexpected format of bins.")

        if frequencies is None:
            self._frequencies = np.zeros(self._bins.shape[0])
        else:
            frequencies = np.array(frequencies, dtype=float)
            if frequencies.shape != (self._bins.shape[0],):
                raise RuntimeError("Values must have same dimension as bins.")
            self._frequencies = frequencies

    @property
    def bins(self):
        """Matrix of bins.

        - 'bin_count' rows
        - 2 columns: left, right edges)."""
        return self._bins

    def __getitem__(self, i):
        """Select sub-histogram or get one bin."""
        if isinstance(i, int):
            return self.bins[i], self.frequencies[i]
        elif isinstance(i, np.ndarray) and i.dtype == bool:
            if i.shape != (self.bin_count,):
                raise IndexError("Cannot index with masked array of a wrong dimension")
        return self.__class__(self.bins[i], self.frequencies[i])

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def densities(self):
        """Frequencies normalized by bin widths.

        Useful when bins are not of the same width.
        """
        return self._frequencies / self.bin_widths

    @property
    def cumulative_frequencies(self):
        return self._frequencies.cumsum()

    @property
    def total(self):
        """Total number (sum of weights) of entries."""
        return self._frequencies.sum()

    def normalize(self, inplace=False):
        if inplace:
            self /= self.total
            return self
        else:
            return self / self.total

    @property
    def numpy_bins(self):
        """Bins in the format of numpy."""
        return np.concatenate((self.bin_left_edges, self.bin_right_edges[-1:]), axis=0)

    @property
    def bin_count(self):
        """Total number of bins."""
        return self.bins.shape[0]

    @property
    def bin_left_edges(self):
        return self.bins[:,0]

    @property
    def bin_right_edges(self):
        return self.bins[:,1]

    @property
    def bin_centers(self):
        return (self.bin_left_edges + self.bin_right_edges) / 2

    @property
    def bin_widths(self):
        return self.bin_right_edges - self.bin_left_edges

    def plot(self, histtype='bar', cumulative=False, normalized=False, backend="matplotlib", axis=None, **kwargs):
        """Plot the histogram.

        :param histtype: ‘bar’ | [‘step’] | 'scatter'
        """
        # TODO: See http://matplotlib.org/1.5.0/examples/api/filled_step.html
        data = self
        if normalized:
            data = data.normalize(inplace=False)
        if cumulative:
            values = data.cumulative_frequencies
        else:
            values = data.densities

        if backend == "matplotlib":
            if not axis:
                import matplotlib.pyplot as plt
                _, axis = plt.subplots()
            # if histtype == "step":
            # TODO: Fix for non-connected histograms
            #     x = np.concatenate(([0.0], self.numpy_bins), axis=0)
            #     y = np.concatenate(([0.0], self.values, [0]), axis=0)
            #     axis.step(x, y, where="post", **kwargs)
            if histtype == "bar":
                axis.bar(self.bin_left_edges, values, self.bin_widths, **kwargs)
            elif histtype == "scatter":
                axis.scatter(self.bin_centers, values, **kwargs)
            else:
                raise RuntimeError("Unknown histogram type: {0}".format(histtype))
        else:
            raise RuntimeError("Only matplotlib supported at the moment.")
        return axis

    def copy(self, include_frequencies=True):
        if include_frequencies:
            frequencies = np.copy(self.frequencies)
        else:
            frequencies = None
        return self.__class__(np.copy(self.bins), frequencies)

    def __eq__(self, other):
        if not isinstance(other, Histogram1D):
            return False
        if not np.array_equal(other.bins, self.bins):
            return False
        if not np.array_equal(other.frequencies, self.frequencies):
            return False
        return True

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __iadd__(self, other):
        if np.isscalar(other):
            raise RuntimeError("Cannot add constant to histograms.")
        if np.allclose(other.bins, self.bins):
            self._frequencies += other.frequencies
        else:
            raise RuntimeError("Bins must be the same when adding histograms.")
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def __isub__(self, other):
        if np.isscalar(other):
            raise RuntimeError("Cannot add constant to histograms.")
        if np.allclose(other.bins, self.bins):
            self._frequencies -= other.frequencies
        else:
            raise RuntimeError("Bins must be the same when subtracting histograms.")
        return self

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        if np.isscalar(other):
            self._frequencies = self._frequencies.astype(float)
            self._frequencies *= other
        else:
            raise RuntimeError("Histograms can be multiplied only by a constant.")
        return self

    def __truediv__(self, other):
        new = self.copy()
        new /= other
        return new

    def __itruediv__(self, other):
        if np.isscalar(other):
            self._frequencies = self._frequencies.astype(float)
            self._frequencies /= other
        else:
            raise RuntimeError("Histograms can be divided only by a constant.")
        return self

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        from collections import OrderedDict
        df = pd.DataFrame({"left": self.bin_left_edges, "right": self.bin_right_edges, "frequency": self.frequencies},
                          columns=["left", "right", "frequency"])
        return df

    def __repr__(self):
        return "{0}(bins={1})".format(
            self.__class__.__name__, self.bins.shape[0]) #, self.total, self.underflow, self.overflow)