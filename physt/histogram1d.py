import numpy as np
from . import bin_utils


class Histogram1D(object):
    """One-dimensional histogram data.

    The bins can be of different widths.

    The bins need not be consecutive. However, some functionality may not be available
    for non-consecutive bins (like keeping information about underflow and overflow).

    Attributes
    ----------
    frequencies: numpy.ndarray
    bins: numpy.ndarray
    errors2: numpy.ndarray
    underflow: float
    name: str
    axis_name: str
    keep_missed: bool

    These are the basic attributes that can be used in the constructor (see there)
    Other attributes are dynamic.
    """
    def __init__(self, bins, frequencies=None, **kwargs):
        """Constructor

        Parameters
        ----------
        bins: array_like
            The bins, either as numpy-like 1D array of edges, or 2D array of edges
        frequencies: Optional[array_like]
            The bin contents.
        keep_missed: Optional[bool]
            Whether to keep track of underflow/overflow when filling with new values.
        underflow: Optional[float]
            Weight of observations that were smaller than the minimum bin.
        overflow: Optional[float]
            Weight of observations that were larger than the maximum bin.
        name: Optional[str]
            Name of the histogram (will be displayed as plot title)
        axis_name: Optional[str]
            Name of the characteristics that is histogrammed (will be displayed on x axis)
        errors2: Optional[array_like]
            Quadratic errors of individual bins. If not set, defaults to frequencies.
        """
        bins = bin_utils.make_bin_array(bins)
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
            if np.any(frequencies < 0):
                raise RuntimeError("Cannot have negative frequencies.")
            self._frequencies = frequencies

        self.keep_missed = kwargs.get("keep_missed", True)
        self.underflow = kwargs.get("underflow", 0)
        self.overflow = kwargs.get("overflow", 0)
        self.name = kwargs.get("name", None)
        self.axis_name = kwargs.get("axis_name", self.name)

        self._errors2 = np.asarray(kwargs.get("errors2", self.frequencies.copy()))
        if np.any(self._errors2 < 0):
            raise  RuntimeError("Cannot have negative squared errors.")

    @property
    def bins(self):
        """Matrix of bins.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array of bin edges, shape=(n, 2)
        """
        return self._bins

    def __getitem__(self, i):
        """Select sub-histogram or get one bin.

        Parameters
        ----------
        i : int or slice or bool masked array or array with indices
            In most cases, this has same semantics as for numpy.ndarray.__getitem__


        Returns
        -------
        Histogram1D or float
            Depending on the parameters, a sub-histogram or content of one bin are returned.
        """
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
        return (self._frequencies / self.bin_widths) / self.total

    @property
    def cumulative_frequencies(self):
        """Cumulative frequencies.

        Note: underflow values are not considered

        Returns
        -------
        numpy.ndarray
        """
        return self._frequencies.cumsum()

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
        """Total number (sum of weights) of entries including underflow and overflow.

        Returns
        -------
        float
        """
        t = self._frequencies.sum()
        if not np.isnan(self.underflow):
            t += self.underflow
        if not np.isnan(self.overflow):
            t += self.overflow
        return t

    @property
    def total_width(self):
        """Total width of all bins.

        In inconsecutive histograms, the missing intervals are not counted in.

        Returns
        -------
        float
        """
        return self.bin_widths.sum()

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
    def numpy_bins(self):
        """Bins in the format of numpy.

        Returns
        -------
        numpy.ndarray
        """
        # TODO: If not consecutive, does not make sense
        return np.concatenate((self.bin_left_edges, self.bin_right_edges[-1:]), axis=0)

    @property
    def bin_count(self):
        """Total number of bins.

        Returns
        -------
        int
        """
        return self.bins.shape[0]

    @property
    def bin_left_edges(self):
        """Left edges of all bins.

        Returns
        -------
        numpy.ndarrad
        """
        return self.bins[:,0]

    @property
    def bin_right_edges(self):
        """Right edges of all bins.

        Returns
        -------
        numpy.ndarray
        """
        return self.bins[:,1]

    @property
    def bin_centers(self):
        """Centers of all bins.

        Returns
        -------
        numpy.ndarray
        """
        return (self.bin_left_edges + self.bin_right_edges) / 2

    @property
    def bin_widths(self):
        """Widths of all bins.

        Returns
        -------
        numpy.ndarray
        """
        return self.bin_right_edges - self.bin_left_edges

    def find_bin(self, value):
        """Index of bin corresponding to a value.

        Parameters
        ----------
        value: float
            Value to be searched for.

        Returns
        -------
        int
            index of bin to which value belongs (-1=underflow, N=overflow, None=not found - inconsecutive)
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

        Parameters
        ----------
        value: float
            Value to be added.
        weight: float, optional
            Weight assigned to the value.

        Returns
        -------
        int
            index of bin which was incremented (-1=underflow, N=overflow, None=not found)

        Note: If a gap in unconsecutive bins is matched, underflow & overflow are not valid anymore.
        Note: Name was selected because of the eponymous method in ROOT
        """
        ixbin = self.find_bin(value)
        if ixbin is None:
            self.overflow = np.nan
            self.underflow = np.nan
        elif ixbin == -1 and self.keep_missed:
            self.underflow += weight
        elif ixbin == self.bin_count and self.keep_missed:
            self.overflow += weight
        else:
            self._frequencies[ixbin] += weight
            self.errors2[ixbin] += weight ** 2
        return ixbin

    def plot(self, histtype='bar', cumulative=False, density=False, errors=False, backend="matplotlib", ax=None, **kwargs):
        """Plot the histogram.

        Parameters
        ----------
        histtype: Optional[str]
            Type of the histogram:
            - bar : as bars, the typical (and default) setting
            - scatter : as points
        cumulative: Optional[bool]
            If True, the values are displayed as cumulative function rising from 0 to N (or N - fraction of unmatched)
        errors: Optional[bool]
            If True, display error bars for bins (not available for cumulative)
        density: Optional[bool]
            If False (default), display absolute values of the bins, if True, display the densities (scaled to bin
            width in standard case, to range 0-1 in the cumulative case).
        backend: str
            Currently, this has to be matplotlib, but other backends (d3.js or bokeh) are planned.
        ax: Optional[matplotlib.axes.Axes]
            The (matplotlib) axes to draw into. If not set, a default one is created.
        figure: Optional[bokeh.plotting.figure.Figure]
            The bokeh figure to draw into.

        You can also specify arbitrary matplotlib arguments, they are forwarded to the respective plotting methods.

        Returns
        -------
        matplotlib.axes.Axes or bokeh.models.plots.Plot
            The axes object for further manipulation.
        """
        # TODO: See http://matplotlib.org/1.5.0/examples/api/filled_step.html
        # TODO: Implement statistics box as in ROOT
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
        else:
            err_data = None

        if backend == "matplotlib":
            if not ax:
                import matplotlib.pyplot as plt
                _, ax = plt.subplots()
            if histtype == "bar":
                bar_kwargs = kwargs.copy()
                if errors:
                    bar_kwargs["yerr"] = err_data
                    if not "ecolor" in bar_kwargs:
                        bar_kwargs["ecolor"] = "black"
                ax.bar(self.bin_left_edges, data, self.bin_widths, **bar_kwargs)
            elif histtype == "scatter":
                if errors:
                    ax.errorbar(self.bin_centers, data, yerr=err_data, fmt=kwargs.get("fmt", "o"), ecolor=kwargs.get("ecolor", "black"))
                else:
                    ax.scatter(self.bin_centers, data, **kwargs)
            else:
                raise RuntimeError("Unknown histogram type: {0}".format(histtype))

            # Automatically limit to positive frequencies
            ylim = ax.get_ylim()
            ylim = (0, max(ylim[1], data.max() + (data.max() - ylim[0]) * 0.1))
            if self.name:
                ax.set_title(self.name)
            if self.axis_name:
                ax.set_xlabel(self.axis_name)
            ax.set_ylim(ylim)
            return ax
        elif backend == "bokeh":
            from bokeh.plotting import figure, output_file, show
            from bokeh.charts import Bar, Scatter, show
            from bokeh.models import HoverTool, Range1d
            from bokeh.models.sources import ColumnDataSource

            if histtype == "scatter":
                plot_data = {
                    "x" : self.bin_centers,
                    "y" : data
                }
                p = Scatter(plot_data, x='x', y='y', **kwargs)
            elif histtype == "bar":
                plot_data = ColumnDataSource(data={
                    "top" : data,
                    "bottom" : np.zeros_like(data),
                    "left" : self.bin_left_edges,
                    "right" : self.bin_right_edges
                })
                tooltips = [
                    ("bin", "@left..@right"),
                    ("frequency", "@top")
                ]
                hover = HoverTool(tooltips=tooltips)
                p = kwargs.get("figure", figure(tools=[hover]))
                p.quad(
                    "left", "right", "top", "bottom",
                    source=plot_data,
                    color=kwargs.get("color", "blue"),
                    line_width=kwargs.get("lw", 1),
                    line_color=kwargs.get("line_color", "black"),
                    fill_alpha=kwargs.get("alpha", 1),
                    line_alpha=kwargs.get("alpha", 1),
                )
            else:
                raise RuntimeError("Unknown histogram type: {0}".format(histtype))
            if kwargs.get("show", True):
                show(p)
            return p
        else:
            raise RuntimeError("Only matplotlib supported at the moment.")

    def copy(self, include_frequencies=True):
        """A deep copy of the histogram.

        Parameters
        ----------
        include_frequencies: Optional[bool]
            If True (default), frequencies are copied. Otherwise, an empty histogram template is created.

        Returns
        -------
        Histogram1D
        """
        if include_frequencies:
            frequencies = np.copy(self.frequencies)
            underflow = self.underflow
            overflow = self.overflow
        else:
            frequencies = None
            underflow = 0
            overflow = 0
        return self.__class__(np.copy(self.bins), frequencies, underflow=underflow, overflow=overflow,
                              name=self.name, axis_name=self.axis_name, keep_missed=self.keep_missed)

    def __eq__(self, other):
        if not isinstance(other, Histogram1D):
            return False
        if not np.allclose(other.bins, self.bins):
            return False
        if not np.allclose(other.frequencies, self.frequencies):
            return False
        if not np.allclose(other.errors2, self.errors2):
            return False
        if not other.overflow == self.overflow:
            return False
        if not other.underflow == self.underflow:
            return False
        if not other.name == self.name:
            return False
        if not other.axis_name == self.axis_name:
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
            self._errors2 += other.errors2
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
            self.underflow -= other.underflow
            self.overflow -= other.overflow
            self._errors2 += other.errors2
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
            self._errors2 *= other
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
            self._errors2 /= other
        else:
            raise RuntimeError("Histograms can be divided only by a constant.")
        return self

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

    def to_dataframe(self):
        """Convert to pandas DataFrame.

        This is not a lossless conversion - (under/over)flow info is lost.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd
        df = pd.DataFrame(
            {
                "left": self.bin_left_edges,
                "right": self.bin_right_edges,
                "frequency": self.frequencies,
                "error": self.errors,
            },
            columns=["left", "right", "frequency", "error"])
        return df

    def to_xarray(self):
        """Convert to xarray.Dataset

        This is an identity conversion.

        Returns
        -------
        xarray.Dataset
        """
        import xarray as xr
        data_vars = {
            "frequencies": xr.DataArray(self.frequencies, dims="bin"),
            "errors2": xr.DataArray(self.errors2, dims="bin"),
            "bins": xr.DataArray(self.bins, dims=("bin", "x01"))
        }
        coords = { }
        attrs = {
            "underflow": self.underflow,
            "overflow": self.overflow,
            "keep_missed": self.keep_missed
        }
        return xr.Dataset(data_vars, coords, attrs)

    @classmethod
    def from_xarray(cls, arr):
        """Convert form xarray.Dataset

        Parameters
        ----------
        arr: xarray.Dataset
            The data in xarray representation
        """
        kwargs = {'frequencies': arr["frequencies"],
                  'bins': arr["bins"],
                  'errors2': arr["errors2"],
                  'overflow': arr.attrs["overflow"],
                  'underflow': arr.attrs["underflow"],
                  'keep_missed': arr.attrs["keep_missed"]}
        return cls(**kwargs)

    def to_json(self, path=None):
        """Convert to JSON representation.

        Parameters
        ----------
        path: Optional[str]
            Where to write the JSON.

        Returns
        -------
        str:
            The JSON representation.
        """
        from collections import OrderedDict
        import json
        data = OrderedDict()
        data["bins"] = self.bins.tolist()
        data["frequencies"] = self.frequencies.tolist()
        data["errors2"] = self.errors2.tolist()
        data["keep_missed"] = self.keep_missed
        data["underflow"] = float(self.underflow)
        data["overflow"] = float(self.overflow)

        text = json.dumps(data)
        if path:
            with open(path, "w", encoding="ascii") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, text=None, path=None):
        """Read histogram from JSON representation.

        Paramaters
        ----------
        text: Optional[str]
            The JSON string itself.
        path: Optional[str]
            Path of the JSON file.

        Returns
        -------
        Histogram1D
        """
        import json
        if text:
            data = json.loads(text)
        else:
            with open(path, "r") as f:
                data = json.load(f)
        return cls(**data)

    def __repr__(self):
        s = "{0}(bins={1}, total={2}".format(
            self.__class__.__name__, self.bins.shape[0], self.total)
        if self.underflow:
            s += ", underflow={0}".format(self.underflow)
        if self.overflow:
            s += ", overflow={0}".format(self.overflow)
        s += ")"
        return s


def calculate_frequencies(data, bins, weights=None, validate_bins=True, already_sorted=False):
    """Get frequencies and bin errors from the data.

    Parameters
    ----------
    data : array_like
        Data items to work on.
    bins : array_like
        A set of bins.
    weights : array_like, optional
        Weights of the items.
    validate_bins : bool, optional
        If True (default), bins are validated to be in ascending order.
    already_sorted : bool, optional
        If True, the data being entered are already sorted, no need to sort them once more.

    Returns
    -------
    frequencies : numpy.ndarray
        Bin contents
    errors2 : numpy.ndarray
        Error squares of the bins
    underflow : float
        Weight of items smaller than the first bin
    overflow : float
        Weight of items larger than the last bin

    Note
    ----
    Checks that the bins are in a correct order (not necessarily consecutive)
    """

    # Ensure correct binning
    bins = bin_utils.make_bin_array(bins)
    if validate_bins:
        if bins.shape[0] == 0:
            raise RuntimeError("Cannot have histogram with 0 bins.")
        if not bin_utils.is_rising(bins):
            raise RuntimeError("Bins must be rising.")

    # Create 1D arrays to work on
    data = np.asarray(data).flatten()
    if weights:
        weights = np.asarray(weights, dtype=float).flatten()
        if weights.shape != data.shape:
            raise RuntimeError("Weight must have the same shape as data")
    else:
        weights = np.ones(data.shape, dtype=int)

    # Data sorting
    if not already_sorted:
        args = np.argsort(data)
        data = data[args]
        weights = weights[args]

    # Fill frequencies and errors
    frequencies = np.zeros(bins.shape[0], dtype=float)
    errors2 = np.zeros(bins.shape[0], dtype=float)
    for xbin, bin in enumerate(bins):
        start = np.searchsorted(data, bin[0], side="left")
        if xbin == 0:
            underflow = weights[0:start].sum()
        if xbin == len(bins) - 1:
            stop = np.searchsorted(data, bin[1], side="right")
            overflow = weights[stop:].sum()
        else:
            stop = np.searchsorted(data, bin[1], side="left")
        frequencies[xbin] = weights[start:stop].sum()
        errors2[xbin] = (weights[start:stop] ** 2).sum()

    # Underflow and overflow don't make sense for unconsecutive binning.
    if not bin_utils.is_consecutive(bins):
        underflow = np.nan
        overflow = np.nan

    return frequencies, errors2, underflow, overflow