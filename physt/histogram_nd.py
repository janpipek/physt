from __future__ import absolute_import
from . import bin_utils
from .histogram_base import HistogramBase
import numpy as np


class HistogramND(HistogramBase):
    """Multi-dimensional histogram data.

    Attributes
    ----------

    """

    def __init__(self, dimension, binnings, frequencies=None, **kwargs):
        # Bins + checks
        if len(binnings) != dimension:
            raise RuntimeError("bins must be a sequence of {0} schemas".format(dimension))

        # self.ndim = dimension
        self._binnings = binnings

        # Frequencies + checks
        if frequencies is None:
            self._frequencies = np.zeros(self.shape)
        else:
            frequencies = np.array(frequencies, dtype=float)
            if frequencies.shape != self.shape:
                raise RuntimeError("Values must have same dimension as bins.")
            if np.any(frequencies < 0):
                raise RuntimeError("Cannot have negative frequencies.")
            self._frequencies = frequencies

        # Missed values
        self.keep_missed = kwargs.get("keep_missed", True)
        self.missed = kwargs.get("missed", 0)

        # Names etc.
        self.name = kwargs.get("name", None)
        self.axis_names = kwargs.get("axis_names", ["axis{0}".format(i) for i in range(self.ndim)])
        if len(self.axis_names) != self.ndim:
            raise RuntimeError("The length of axis names must be equal to histogram dimension.")

        # Errors + checks
        self._errors2 = kwargs.get("errors2")
        if self._errors2 is None:
            self._errors2 = self._frequencies.copy()
        else:
            self._errors2 = np.asarray(self._errors2)
        if self._errors2.shape != self._frequencies.shape:
            raise RuntimeError("Errors must have same dimension as frequencies.")
        if np.any(self._errors2 < 0):
            raise RuntimeError("Cannot have negative squared errors.")

    @property
    def bins(self):
        """Matrix of bins.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array of bin edges, shape=(n, 2)
        """
        return [binning.bins for binning in self._binnings]

    @property
    def numpy_bins(self):
        return [binning.numpy_bins for binning in self._binnings]  

    def __getitem__(self, item):
        raise NotImplementedError()

    # Missing: cumulative_frequencies - does it make sense

    def get_bin_widths(self, axis = None):  # -> Base
        if axis is not None:
            return self.get_bin_right_edges(axis) - self.get_bin_left_edges(axis)
        else:
            return np.meshgrid(*[self.get_bin_widths(i) for i in range(self.ndim)], indexing='ij')

    @property
    def bin_sizes(self):
        # TODO: Some kind of caching?
        sizes = self.get_bin_widths(0)
        for i in range(1, self.ndim):
            sizes = np.outer(sizes, self.get_bin_widths(i))
        return sizes

    @property
    def total_size(self):
        """The default size of the bin space."""
        return np.product([self.get_bin_widths(i) for i in range(self.ndim)])

    def get_bin_left_edges(self, axis=None):
        if axis is not None:
            return self.bins[axis][:, 0]
        else:
            return np.meshgrid(*[self.get_bin_left_edges(i) for i in range(self.ndim)], indexing='ij')

    def get_bin_right_edges(self, axis=None):
        if axis is not None:
            return self.bins[axis][:, 1]
        else:
            return np.meshgrid(*[self.get_bin_right_edges(i) for i in range(self.ndim)], indexing='ij')

    def get_bin_centers(self, axis=None):
        if axis is not None:
            return (self.get_bin_right_edges(axis) + self.get_bin_left_edges(axis)) / 2
        else:
            return np.meshgrid(*[self.get_bin_centers(i) for i in range(self.ndim)], indexing='ij')

    def find_bin(self, value, axis=None):  # TODO: Check!
        if axis is not None:
            ixbin = np.searchsorted(self.get_bin_left_edges(axis), value, side="right")
            if ixbin == 0:
                return None
            elif ixbin == self.shape[axis]:
                if value <= self.get_bin_right_edges(axis)[-1]:
                    return ixbin - 1
                else:
                    return None
            elif value < self.get_bin_right_edges(axis)[ixbin - 1]:
                return ixbin - 1
            elif ixbin == self.shape[axis]:
                return None
            else:
                return None
        else:
            ixbin = tuple(self.find_bin(value[i], i) for i in range(self.ndim))
            if None in ixbin:
                return None
            else:
                return ixbin 

    def fill(self, value, weight=1):
        for i, binning in enumerate(self._binnings):
            if binning.is_adaptive():
                #print("adaptive, forcing", value[i])
                bin_map = binning.force_bin_existence(value[i])
                #print("map", bin_map)
                self._reshape_data(binning.bin_count, bin_map, i)

        ixbin = self.find_bin(value)
        if ixbin is None and self.keep_missed:
            self.missed += weight
        else:
            self._frequencies[ixbin] += weight
            self.errors2[ixbin] += weight ** 2
        return ixbin

    def fill_n(self, values, weights=None, dropna=True):
        values = np.asarray(values)
        if dropna:
            values = values[~np.isnan(values)]
        for i, binning in enumerate(self._binnings):
            if binning.is_adaptive():
                map = self._binning.force_bin_existence(values[:,i])
                self._reshape_data(binning.bin_count, map, i)
        frequencies, errors2, underflow, overflow, stats = calculate_frequencies(values, self.ndim, self.bins,
                                                                              weights=weights)
        self._frequencies += frequencies
        self._errors2 += errors2
        # TODO: check that adaptive does not produce under-/over-flows?
        self.underflow += underflow
        self.overflow += overflow
        for key in self._stats:
            self._stats[key] += stats.get(key, 0.0)    

    def copy(self, include_frequencies=True):
        if include_frequencies:
            frequencies = np.copy(self.frequencies)
            missed = self.missed
            errors2 = np.copy(self.errors2)
        else:
            frequencies = None
            missed = 0
            errors2 = None
        return self.__class__(dimension=self.ndim, binnings=[binning.copy() for binning in self._binnings],
                              frequencies=frequencies, errors2=errors2,
                              name=self.name, axis_names=self.axis_names[:],
                              keep_missed=self.keep_missed, missed=missed)

    def projection(self, *axes, **kwargs):
        axes = list(axes)
        for i, ax in enumerate(axes):
            if isinstance(ax, str):
                if not ax in self.axis_names:
                    raise RuntimeError("Invalid axis name for projection: " + ax)
                axes[i] = self.axis_names.index(ax)
        if not axes:
            raise RuntimeError("No axis selected for projection")
        if len(axes) != len(set(axes)):
            raise RuntimeError("Duplicate axes in projection")
        invert = list(range(self.ndim))
        for ax in axes:
            invert.remove(ax)
        axes = tuple(axes)
        invert = tuple(invert)
        frequencies = self.frequencies.sum(axis=invert)
        errors2 = self.errors2.sum(axis=invert)
        name = kwargs.pop("name", self.name)
        axis_names = [name for i, name in enumerate(self.axis_names) if i in axes]
        bins = [bins for i, bins in enumerate(self._binnings) if i in axes]
        if len(axes) == 1:
            from .histogram1d import Histogram1D
            return Histogram1D(binning=bins[0], frequencies=frequencies, errors2=errors2,
                               axis_name=axis_names[0], name=name)
        elif len(axes) == 2:
            return Histogram2D(binnings=bins, frequencies=frequencies, errors2=errors2,
                               axis_names=axis_names, name=name)
        else:
            return HistogramND(dimension=len(axes), binnings=bins, frequencies=frequencies, errors2=errors2,
                               axis_names=axis_names, name=name)

    def __eq__(self, other):
        """Equality comparison

        """
        # TODO: Describe allclose
        # TODO: Think about softer alternatives (like compare method)
        if not isinstance(other, self.__class__):
            return False
        if not self.ndim == other.ndim:
            return False
        for i in range(self.ndim):
            if not np.allclose(other.bins[i], self.bins[i]):
                return False
        if not np.allclose(other.errors2, self.errors2):
            return False
        if not np.allclose(other.frequencies, self.frequencies):
            return False
        if not other.missed == self.missed:
            return False
        if not other.name == self.name:
            return False
        if not other.axis_names == self.axis_names:
            return False
        return True

    def __iadd__(self, other):
        if np.isscalar(other):
            raise RuntimeError("Cannot add constant to histograms.")        
        if other.ndim != self.ndim:
                raise RuntimeError("Cannot add histograms with different dimensions.")            
        elif self.has_same_bins(other):
            # print("Has same!!!!!!!!!!")
            self._frequencies += other.frequencies
            self._errors2 += other.errors2 
            self.missed += other.missed
        elif self.is_adaptive():
            if other.missed > 0:
                raise RuntimeError("Cannot adapt histogram with missed values.")
            try:
                other = other.copy()
                other.set_adaptive(True)

                # TODO: Fix state after exception
                # maps1 = []
                maps2 = []
                for i in range(self.ndim):
                    new_bins = self._binnings[i].copy()

                    map1, map2 = new_bins.adapt(other._binnings[i])
                    self.change_binning(new_bins, map1, axis=i)
                    other.change_binning(new_bins, map2, axis=i)
                self._frequencies += other.frequencies
                self._errors2 += other.errors2 

            except:
                raise # RuntimeError("Cannot find common binning for added histograms.")           
        else:
            raise RuntimeError("Incompatible binning")
        return self

    def __imul__(self, other):
        if not np.isscalar(other):
            raise RuntimeError("Histograms may be multiplied only by a constant.")  
        self._frequencies *= other
        self._errors2 *= other ** 2
        self.missed *= other
        return self

    def __isub__(self, other):
        return self.__iadd__(other * (-1))

    def __itruediv__(self, other):
        if not np.isscalar(other):
            raise RuntimeError("Histograms may be divided only by a constant.")  
        self._frequencies /= other
        self._errors2 /= other ** 2
        self.missed /= other
        return self

    # def to_xarray(self):
    #     raise NotImplementedError()

    # @classmethod
    # def from_xarray(cls, arr):
    #     raise  NotImplementedError()

    # def to_json(self, path=None):
    #     raise NotImplementedError()

    # @classmethod
    # def from_json(cls, text=None, path=None):
    #     return NotImplementedError

    def __repr__(self):
        s = "{0}(bins={1}, total={2}".format(
            self.__class__.__name__, self.shape, self.total)
        if self.missed:
            s += ", missed={0}".format(self.missed)
        s += ")"
        return s


class Histogram2D(HistogramND):
    """Specialized 2D variant of the general HistogramND class.

    Its only addition is the plot() method
    """

    def __init__(self, binnings, frequencies=None, **kwargs):
        kwargs.pop("dimension", None)
        super(Histogram2D, self).__init__(dimension=2, binnings=binnings, frequencies=frequencies, **kwargs)

    def plot(self, histtype='map', density=False, backend="matplotlib", ax=None, **kwargs):
        """Plot the 2D histogram.
        
        Parameters
        ----------
        histtype: str
            map or bar3d
        cmap: str or matplotlib.colors.Colormap
            The colormap of name of the colormap (used in map)
        cmap_min: float or str
            Minimum value to include in the colormap (lower are clipped, default: 0)
            Special value: "min" -> Minimum value is the minimum bin value
        cmap_max: float
            Maximum value to include in the colormap (higher are clipped, default: maximum bin value)            
        show_zero: bool
            Draw bins with zero frequency (default: True)
        show_values: bool
            Show little labels with bin values (default: False)
        format_value: function
            The thing to be displayed as value (useful example: int)            
        show_colorbar: bool
            Display a colobar with range on the right of the axis
        grid_color: str
            Color of the grid in the colour map (default: middle of the colormap)
        alpha: Optional[bool]
            0.0..1.0 of the whole plot (not fully implemented, default: 1.0)
        lw: float
            Width of the grid lines
                
        """
        color = kwargs.pop("color", "frequency")
        alpha = kwargs.pop("alpha", 1.0)
        show_zero = kwargs.pop("show_zero", True)
        show_colorbar = kwargs.pop("show_colorbar", True)
        show_values = kwargs.pop("show_values", False)
        format_value = kwargs.pop("format_value", str)

        if density:
            dz = self.densities.flatten()
        else:
            dz = self.frequencies.flatten()

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.colors as colors

            if color == "frequency":
                cmap = kwargs.pop("cmap", "Greys")
                cmap_max = kwargs.pop("cmap_max", dz.max())
                cmap_min = kwargs.pop("cmap_min", 0)
                if cmap_min == "min":
                    cmap_min = dz.min()
                norm = colors.Normalize(cmap_min, cmap_max, clip=True)
                
                if isinstance(cmap, str):
                    cmap = plt.get_cmap(cmap)
                colors = cmap(norm(dz))
            else:
                colors = color   # TODO: does not work for map

            if histtype == "bar3d":
                from mpl_toolkits.mplot3d import Axes3D
                xpos, ypos = (arr.flatten() for arr in self.get_bin_centers())
                zpos = np.zeros_like(ypos)
                dx, dy = (arr.flatten() for arr in self.get_bin_widths())

                if not ax:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
                ax.set_xlabel(self.axis_names[0])
                ax.set_ylabel(self.axis_names[1])
                ax.set_zlabel("density" if density else "frequency")
                if self.name:
                    ax.set_title(self.name)
            elif histtype == "map":
                if not ax:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                else:
                    fig = ax.get_figure()

                xpos, ypos = (arr.flatten() for arr in self.get_bin_left_edges())
                if show_values:
                    text_x, text_y = (arr.flatten() for arr in self.get_bin_centers())
                
                dx, dy = (arr.flatten() for arr in self.get_bin_widths())

                for i in range(len(xpos)):
                    bin_color = colors[i]
                
                    if dz[i] > 0 or show_zero:
                        rect = plt.Rectangle([xpos[i], ypos[i]], dx[i], dy[i],
                                            facecolor=bin_color, edgecolor=kwargs.get("grid_color", cmap(0.5)),
                                            lw=kwargs.get("lw", 0.5), alpha=alpha)
                        ax.add_patch(rect)
                        
                        if show_values:
                            text = format_value(dz[i])
                            yiq_y = np.dot(bin_color[:3], [0.299, 0.587, 0.114])
                                
                            if yiq_y > 0.5:
                                text_color = (0.0, 0.0, 0.0, alpha)
                            else:
                                text_color = (1.0, 1.0, 1.0, alpha)
                            ax.text(text_x[i], text_y[i], text, horizontalalignment='center', verticalalignment='center', color=text_color)                        
                                                
                ax.set_xlim(self.bins[0][0,0], self.bins[0][-1,1])
                ax.set_ylim(self.bins[1][0,0], self.bins[1][-1,1])
                ax.autoscale_view()
                ax.set_xlabel(self.axis_names[0])
                ax.set_ylabel(self.axis_names[1])
                if self.name:
                    ax.set_title(self.name)

                if show_colorbar:
                    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
                    mappable.set_array(dz)

                    fig.colorbar(mappable, ax=ax)
                fig.tight_layout()
            else:
                raise RuntimeError("Unsupported hist type")
            return ax
        else:
            raise RuntimeError("Unsupported hist type")


def calculate_frequencies(data, ndim, binnings, weights=None):
    data = np.asarray(data)

    edges_and_mask = [binning.numpy_bins_with_mask for binning in binnings]
    edges = [em[0] for em in edges_and_mask]
    masks = [em[1] for em in edges_and_mask]

    ixgrid = np.ix_(*masks) # Indexer to select parts we want

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != (data.shape[0],):
            raise RuntimeError("Invalid weights shape.")
        total_weight = weights.sum()
    else:
        total_weight = data.shape[0]

    # TODO: Right edges are not taken into account because they fall into inf bin

    if data.shape[0]:
        frequencies, _ = np.histogramdd(data, edges, weights=weights)
        frequencies = frequencies[ixgrid].copy()
        missing = total_weight - frequencies.sum()
        if weights is not None:
            err_freq, _ = np.histogramdd(data, edges, weights=weights ** 2)
            errors2 = err_freq[ixgrid].copy()
        else:
            errors2 = frequencies.copy()        
    else:
        frequencies = None
        missing = 0
        errors2 = None

    return frequencies, errors2, missing

