import numpy as np


class HistogramBase(object):
    """Behaviour shared between all histogram classes."""

    # @property
    # def bins(self):
    #     Matrix of bins.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         Two-dimensional array of bin edges, shape=(n, 2)
        
    #     return [binning.bins for binning in self._binnings]

    # @property
    # def numpy_bins(self):
    #     return [binning.numpy_bins for binning in self._binnings]

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

    def has_same_bins(self, other):
        """Whether two histogram share the same binning.

        Returns
        -------
        bool
        """
        if self.shape != other.shape:
            return False                 
        elif self.ndim == 1:
            return np.allclose(self.bins, other.bins)
        elif self.ndim > 1:
            for i in range(self.ndim):
                if not np.allclose(self.bins[i], other.bins[i]):
                    return False
            return True        

    def has_compatible_bins(self, other):
        # By default, the bins must be the same
        # Overridden
        return self.has_same_bins()

    def fill_n(self, values, weights=None):
        if weights is not None:
            if weights.shape != values.shape[0]:
                raise RuntimeError("Wrong shape of weights")
        for i, value in enumerate(values):
            if weights is not None:
                self.fill(value, weights[i])
            else:
                self.fill(value)

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __radd__(self, other):
        if other == 0:    # Enable sum()
            return self
        else:
            return self + other        

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

