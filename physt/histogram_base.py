from __future__ import absolute_import
import numpy as np


class HistogramBase(object):
    """Behaviour shared between all histogram classes.

    The most important daughter classes are:
    - Histogram1D
    - HistogramND   

    Attributes
    ----------
    _binnings : Iterable[BinningBase]
        Schema for binning(s)
    _frequencies : array_like
        Bin contents
    _errors2 : array_like
        Square errors associated with the bin contents
    """

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
    def shape(self):
        """Shape of histogram's data.

        Returns
        -------
        tuple[int]
            One-element tuple
        """        
        return tuple(bins.bin_count for bins in self._binnings)

    @property
    def ndim(self):
        """Dimensionality of histogram's data.

        i.e. the number of axes along which we bin the values.

        Returns
        -------
        int
        """
        return len(self._binnings)


    @property
    def bin_count(self):
        """Total number of bins.

        Returns
        -------
        int
        """
        return np.product(self.shape)        

    @property
    def frequencies(self):
        """Frequencies (values) of the histogram.

        Returns
        -------
        numpy.ndarray
            Array of bin frequencies
        """
        return self._frequencies

    @property
    def densities(self):
        """Frequencies normalized by bin sizes.

        Useful when bins are not of the same size.

        Returns
        -------
        numpy.ndarray
        """
        return (self._frequencies / self.bin_sizes)

    def normalize(self, inplace=False):
        """Normalize the histogram, so that the total weight is equal to 1.

        See also
        --------
        densities

        """
        if inplace:
            self /= self.total
            return self
        else:
            return self / self.total        

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

    def is_adaptive(self):
        """Whether the binning can be changed with operations.

        Returns
        -------
        bool
        """
        return all(binning.is_adaptive() for binning in self._binnings)

    def set_adaptive(self, value=True):
        for binning in self._binnings:
            binning.set_adaptive(value)

    def change_binning(self, new_binning, bin_map, axis=0):
        """Set new binnning and update the bin contents according to a map.

        Fills frequencies and errors with 0.
        It's the caller's responsibility to provide correct binning and map.

        Parameters
        ----------
        new_binning: physt.binnings.BinningBase
        bin_map: Iterable[tuple]
            tuples contain bin indices (old, new)
        axis: int
            What axis does the binning describe(0..ndim-1)
        """
        axis = int(axis)
        if axis < 0 or axis >= self.ndim:
            raise RuntimeError("Axis must be in range 0..(ndim-1)")
        self._reshape_data(new_binning.bin_count, bin_map, axis)
        self._binnings[axis] = new_binning    

    def merge_bins(self, amount=None, min_frequency=None, axis=None, inplace=True):
        """Reduce the number of bins and add their content:

        Parameters
        ----------
        amount: int
            How many adjacent bins to join together.
        min_frequency: float
            Try to have at least this value in each bin
            (this is not enforce e.g. for minima between high bins)
        axis: int or None
            On which axis to do this (None => all)
        inplace:
            Whether to modify this histogram or return a new one

        Returns
        -------
        HistogramBase or None
            if inplace, return
        """
        if not inplace:
            histogram = self.copy()
            histogram.merge_bins(amount, min_frequency=min_frequency, axis=axis)
            return histogram
        elif axis is None:
            for i in range(self.ndim):
                self.merge_bins(amount=amount, min_frequency=min_frequency, axis=i)
        else:
            if amount is not None:
                if not amount == int(amount):
                    raise RuntimeError("Amount must be integer")
                bin_map = [(i, i // amount) for i in range(self.shape[axis])]    
            elif min_frequency is not None:
                if self.ndim == 1:
                    check = self.frequencies
                else:
                    check = self.projection(axis).frequencies
                bin_map = []
                current_new = 0
                current_sum = 0
                for i, freq in enumerate(check):
                    if freq >= min_frequency and current_sum > 0:
                        current_sum = 0
                        current_new += 1
                    bin_map.append((i, current_new))
                    current_sum += freq
                    if current_sum > min_frequency:
                        current_sum = 0
                        current_new += 1
            else:
                raise NotImplementedError("Not yet implemented.")
            new_binning = self._binnings[axis].apply_bin_map(bin_map)
            self.change_binning(new_binning, bin_map, axis=axis)

    def _reshape_data(self, new_size, bin_map, axis=0):
        """Reshape data to match new binning schema.

        Fills frequencies and errors with 0.

        Parameters
        ----------
        new_size: int
        bin_map: Iterable[(old, new)] or int or None
            If none, we can keep the data unchanged.
        axis:
            On which axis to apply        
        """
        if bin_map is None:    
            return
        else:
            new_shape = list(self.shape)
            new_shape[axis] = new_size            
            new_frequencies = np.zeros(new_shape, dtype=float)
            new_errors2 = np.zeros(new_shape, dtype=float)
            self._apply_bin_map(
                old_frequencies=self._frequencies, new_frequencies=new_frequencies,
                old_errors2=self._errors2, new_errors2=new_errors2,
                bin_map=bin_map, axis=axis)                
            self._frequencies = new_frequencies
            self._errors2 = new_errors2    

    def _apply_bin_map(self, old_frequencies, new_frequencies, old_errors2, new_errors2, bin_map, axis=0):
        if old_frequencies is not None and old_frequencies.shape[axis] > 0:
            if isinstance(bin_map, int):
                new_index = [slice(None) for i in range(self.ndim)]
                new_index[axis] = slice(bin_map, bin_map + old_frequencies.shape[axis])          
                new_frequencies[new_index] += old_frequencies
                new_errors2[new_index] += old_errors2
            else:
                for (old, new) in bin_map:      # Generic enough
                    new_index = [slice(None) for i in range(self.ndim)]
                    new_index[axis] = new
                    old_index = [slice(None) for i in range(self.ndim)]
                    old_index[axis] = old
                    new_frequencies[new_index] += old_frequencies[old_index]
                    new_errors2[new_index] += old_errors2[old_index]       

    def has_same_bins(self, other):
        """Whether two histograms share the same binning.

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

