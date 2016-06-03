import numpy as np


class HistogramBase(object):
    """Behaviour shared between all histogram classes."""

    @property
    def bins(self):
        """Matrix of bins.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array of bin edges, shape=(n, 2)
        """
        return self._bins

    @property
    def frequencies(self):
        """Frequencies (values) of the histogram.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of bin frequencies
        """
        return self._frequencies

    @property
    def densities(self):
        """Frequencies normalized by bin widths.

        Useful when bins are not of the same width.

        Returns
        -------
        numpy.ndarray
        """
        return (self._frequencies / self.bin_sizes) / self.total

    @property
    def errors2(self):
        """Squares of the bin errors.

        Returns
        -------
        numpy.ndarray
        """
        return self._errors2

    @property
    def errors(self):
        """Bin errors

        Returns
        -------
        numpy.ndarray
        """
        return np.sqrt(self.errors2)

    @property
    def total(self):
        """Total number (sum of weights) of entries excluding underflow and overflow.

        Returns
        -------
        float
        """
        return self._frequencies.sum()

    def same_bins(self, other):
        """Whether two histogram share the same binning.

        Returns
        -------
        bool
        """
        if self.ndim != other.ndim:
            return False
        elif self.ndim == 1:
            return np.allclose(self.bins, other.bins)
        elif self.ndim > 1:
            for i in range(self.ndim):
                if not np.allclose(self.bins[i], other.bins[i]):
                    return False
                return True

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        new = self.copy()
        new /= other
        return new

    def __array__(self):
        """Convert to numpy array.

        Returns
        -------
        numpy.ndarray
            The array of frequencies

        See also
        --------
        frequencies
        """
        return self.frequencies

