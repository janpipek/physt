import numpy as np


class Histogram1D(object):
    """Representation of one-dimensional histogram.
    """
    def __init__(self, bins, frequencies=None, **kwargs):
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

        self.underflow = kwargs.get("underflow", 0)
        self.overflow = kwargs.get("overflow", 0)

        self.errors2 = kwargs.get("errors2", None)

        # TODO: if bins are not consecutive, overflow and underflow don't make any sense

    @property
    def bins(self):
        """Matrix of bins.

        - 'bin_count' rows
        - 2 columns: left, right edges)."""
        return self._bins

    def __getitem__(self, i):
        """Select sub-histogram or get one bin."""
        underflow = np.nan
        overflow = np.nan
        if isinstance(i, int):
            return self.bins[i], self.frequencies[i]
        elif isinstance(i, np.ndarray):
            if i.dtype == bool:
                if i.shape != (self.bin_count,):
                    raise IndexError("Cannot index with masked array of a wrong dimension")
        elif isinstance(i, slice):
            if i.step:
                raise IndexError("Cannot change the order of bins")
            if i.step == 1 or i.step is None:
                underflow = self.underflow
                overflow = self.overflow
                if i.start:
                    underflow += self.frequencies[0:i.start].sum()
                if i.stop:
                    overflow += self.frequencies[i.stop:].sum()
        return self.__class__(self.bins[i], self.frequencies[i], overflow=overflow, underflow=underflow)

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def densities(self):
        """Frequencies normalized by bin widths.

        Useful when bins are not of the same width.
        """
        return (self._frequencies / self.bin_widths) / self.total

    @property
    def cumulative_frequencies(self):
        """

        Note: underflow values are not considered
        """
        return self._frequencies.cumsum()

    @property
    def errors(self):
        if self.errors2:
            return np.sqrt(self.errors2)
        else:
            return np.sqrt(self.frequencies)

    @property
    def total(self):
        """Total number (sum of weights) of entries including underflow and overflow."""
        t = self._frequencies.sum()
        if not np.isnan(self.underflow):
            t += self.underflow
        if not np.isnan(self.overflow):
            t += self.overflow
        return t

    @property
    def total_width(self):
        return self.bin_widths.sum()

    def normalize(self, inplace=False):
        if inplace:
            self /= self.total
            return self
        else:
            return self / self.total

    @property
    def numpy_bins(self):
        """Bins in the format of numpy."""
        # TODO: If not consecutive, does not make sense
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

    def find_bin(self, value):
        """Index of bin corresponding to a value.

        :return index, (0=underflow, N=overflow, None=not found)

        Bins are defined as [left, right)
        """
        ixbin = np.searchsorted(self.bin_left_edges, value, side="right")
        if ixbin == 0:
            return -1
        elif ixbin == self.bin_count:
            if value <= self.bin_right_edges[-1]:
                return ixbin - 1
            else:
                return self.bin_count
        elif value < self.bin_right_edges[ixbin - 1]:
            return ixbin - 1
        elif ixbin == self.bin_count:
            return self.bin_count
        else:
            return None

    def fill(self, value, weight=1):
        """Update histogram with a new value.

        :return index of bin which was incremented (0=underflow, N=overflow, None=not found)
        """
        ixbin = self.find_bin(value)
        if ixbin is None:
            pass # or error
        elif ixbin == -1:
            self.underflow += weight
        elif ixbin == self.bin_count:
            self.overflow += weight
        else:
            self._frequencies[ixbin] += weight
        return ixbin

    @property
    def bin_widths(self):
        return self.bin_right_edges - self.bin_left_edges

    def plot(self, histtype='bar', cumulative=False, density=False, errors=False, backend="matplotlib", axis=None, **kwargs):
        """Plot the histogram.

        :param histtype: ‘bar’ | 'scatter'
        """
        # TODO: See http://matplotlib.org/1.5.0/examples/api/filled_step.html
        data = self
        if density:
            if cumulative:
                data = (self / self.total).cumulative_frequencies
            else:
                data = self.densities
        else:
            if cumulative:
                data = self.cumulative_frequencies
            else:
                data = self.frequencies
        if errors:
            if cumulative:
                raise NotImplementedError("Errors not implemented for cumulative plots.")
            if density:
                err_data = self.densities / self.frequencies * self.errors
            else:
                err_data = self.errors

        if backend == "matplotlib":
            if not axis:
                import matplotlib.pyplot as plt
                _, axis = plt.subplots()
            if histtype == "bar":
                bar_kwargs = kwargs.copy()
                if errors:
                    bar_kwargs["yerr"] = err_data
                    if not "ecolor" in bar_kwargs:
                        bar_kwargs["ecolor"] = "black"
                axis.bar(self.bin_left_edges, data, self.bin_widths, **bar_kwargs)
            elif histtype == "scatter":
                if errors:
                    axis.errorbar(self.bin_centers, data, yerr=err_data, fmt=kwargs.get("fmt", "o"), ecolor=kwargs.get("ecolor", "black"))
                else:
                    axis.scatter(self.bin_centers, data, **kwargs)
            else:
                raise RuntimeError("Unknown histogram type: {0}".format(histtype))

            # Automatically limit to positive frequencies
            ylim = axis.get_ylim()
            ylim = (0, max(ylim[1], data.max() + (data.max() - ylim[0]) * 0.1))
            axis.set_ylim(ylim)
        else:
            raise RuntimeError("Only matplotlib supported at the moment.")

        return axis

    def copy(self, include_frequencies=True):
        if include_frequencies:
            frequencies = np.copy(self.frequencies)
            underflow = self.underflow
            overflow = self.overflow
        else:
            frequencies = None
            underflow = 0
            overflow = 0
        return self.__class__(np.copy(self.bins), frequencies, underflow=underflow, overflow=overflow)

    def __eq__(self, other):
        if not isinstance(other, Histogram1D):
            return False
        if not np.array_equal(other.bins, self.bins):
            return False
        if not np.array_equal(other.frequencies, self.frequencies):
            return False
        if not other.overflow == self.overflow:
            return False
        if not other.underflow == self.underflow:
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
            self.underflow += other.underflow
            self.overflow += other.overflow
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
            self.underflow += other.underflow
            self.overflow += other.overflow
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
            self.overflow *= other
            self.underflow *= other
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
            self.overflow /= other
            self.underflow /= other
        else:
            raise RuntimeError("Histograms can be divided only by a constant.")
        return self

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        # TODO Include underflow, overflow
        import pandas as pd
        from collections import OrderedDict
        df = pd.DataFrame({"left": self.bin_left_edges, "right": self.bin_right_edges, "frequency": self.frequencies},
                          columns=["left", "right", "frequency"])
        return df

    def __repr__(self):
        s = "{0}(bins={1}, total={2}".format(
            self.__class__.__name__, self.bins.shape[0], self.total)
        if self.underflow:
            s += ", underflow={0}".format(self.underflow)
        if self.overflow:
            s += ", overflow={0}".format(self.overflow)
        s += ")"
        return s