from __future__ import absolute_import
from .histogram_base import HistogramBase
from .histogram_nd import HistogramND
from .histogram1d import Histogram1D
from . import binnings, histogram_nd
import numpy as np
import math
from functools import reduce


class TransformedHistogramMixin(object):
    """Histogram with non-cartesian (or otherwise transformed) axes.

    This is a mixin, providing transform-aware find_bin, fill and fill_n.

    When implementing, you are required to provide tbe following:
    - `transform` method to convert rectangular (suggested to make it classmethod)
    - `bin_sizes` property

    In certain cases, you may want to have default axis names + projections.
    """

    @classmethod
    def transform(self, value):
        """Convert cartesian coordinates into internal ones.

        Parameters
        ----------
        value : array_like
            This method should accept both scalars and numpy arrays.

        Returns
        -------
        float or array_like
        """
        raise NotImplementedError("TransformedHistogramMixin descendant must implement transform method.")

    def find_bin(self, value, axis=None, transformed=False):
        """

        Parameters
        ----------
        value : array_like
            Value with dimensionality equal to histogram.
        transformed : bool
            If true, the value is already transformed and has same axes as the bins.
        """
        if axis is None and not transformed:
            value = self.transform(value)
        return HistogramND.find_bin(self, value, axis=axis)

    @property
    def bin_sizes(self):
        raise NotImplementedError("TransformedHistogramMixin descendant must implement bin_sizes property.")

    def fill(self, value, weight=1, transformed=False):
        return HistogramND.fill(self, value=value, weight=weight, transformed=transformed)

    def fill_n(self, values, weights=None, dropna=True, transformed=False):
        if not transformed:
            values = self.transform(values)
        HistogramND.fill_n(self, values=values, weights=weights, dropna=dropna)


class PolarHistogram(TransformedHistogramMixin, HistogramND):
    """Polar histogram data.

    This is a special case of a 2D histogram with transformed coordinates.
    """
    def __init__(self, binnings, frequencies=None, **kwargs):
        if not "axis_names" in kwargs:
            kwargs["axis_names"] = ("r", "phi")
        if "dim" in kwargs:
            kwargs.pop("dim")
        super(PolarHistogram, self).__init__(2, binnings=binnings, frequencies=frequencies, **kwargs)

    @property
    def bin_sizes(self):
        sizes = 0.5 * (self.get_bin_right_edges(0) ** 2 - self.get_bin_left_edges(0) ** 2)
        sizes = np.outer(sizes, self.get_bin_widths(1))
        return sizes

    @classmethod
    def transform(self, value):
        r = np.hypot(value[1], value[0])
        phi = np.arctan2(value[1], value[0]) % (2 * np.pi)
        return (r, phi)

    def projection(self, axis_name, **kwargs):
        if isinstance(axis_name, int):
            ax = axis_name
        elif axis_name == self.axis_names[0]:
            ax = 0
        elif axis_name == self.axis_names[1]:
            ax = 1
        else:
            raise RuntimeError("Unknown axis: {0}".format(axis_name))
        klass = (RadialHistogram, AzimuthalHistogram)[ax]
        return HistogramND.projection(self, ax, type=klass, **kwargs)

    # TODO: fill_n() does not work


class RadialHistogram(Histogram1D):
    """Projection of polar histogram to 1D with respect to radius.

    This is a special case of a 1D histogram with transformed coordinates.
    """
    @property
    def bin_sizes(self):
        return (self.bin_right_edges ** 2 - self.bin_left_edges ** 2) * np.pi

    def fill_n(self, values, weights=None, dropna=True):
        # TODO: Implement?
        raise NotImplementedError("Radial histogram is not (yet) modifiable")

    def fill(self, value, weight=1):
        # TODO: Implement?
        raise NotImplementedError("Radial histogram is not (yet) modifiable")


class AzimuthalHistogram(Histogram1D):
    """Projection of polar histogram to 1D with respect to phi.

    This is a special case of a 1D histogram with transformed coordinates.
    """
    # TODO: What about fill(_n)? Should it be 1D or 2D?
    # TODO: Add special plotting (polar bar, polar ring)
    def fill_n(self, values, weights=None, dropna=True):
        raise NotImplementedError("Azimuthal histogram is not (yet) modifiable")

    def fill(self, value, weight=1):
        raise NotImplementedError("Azimuthal histogram is not (yet) modifiable")


class SphericalHistogram(TransformedHistogramMixin, HistogramND):
    def __init__(self, binnings, frequencies=None, **kwargs):
        if not "axis_names" in kwargs:
            kwargs["axis_names"] = ("r", "theta", "phi")
        kwargs.pop("dim", False)
        super(SphericalHistogram, self).__init__(3, binnings=binnings, frequencies=frequencies, **kwargs)

    @classmethod
    def transform(self, value):
        x, y, z = value
        xy = np.hypot(x, y)
        r = np.hypot(xy, z)
        theta = np.arctan2(xy, z) % (2 * np.pi)
        phi = np.arctan2(x, y) % (2 * np.pi)
        return (r, theta, phi)

    @property
    def bin_sizes(self):
        sizes = 0.5 * (self.get_bin_right_edges(0) ** 2 - self.get_bin_left_edges(0) ** 2)
        sizes2 = np.cos(self.get_bin_left_edges(1)) - np.cos(self.get_bin_right_edges(1))               # Hopefully correct
        return reduce(np.multiply, np.ix_(sizes, sizes2, self.get_bin_widths(2)))
        #return np.outer(sizes, sizes2, self.get_bin_widths(2))    # Correct


def polar_histogram(xdata, ydata, radial_bins="human", phi_bins=16, transformed=False, *args, **kwargs):
    """

    Parameters
    ----------
    transformed : bool
    phi_range : Optional[tuple]
    range
    """
    if not transformed:
        rdata = np.hypot(ydata, xdata)
        phidata = np.arctan2(ydata, xdata) % (2 * np.pi)
    else:
        rdata = xdata
        phidata = ydata
    data = np.concatenate([rdata[:, np.newaxis], phidata[:, np.newaxis]], axis=1)
    dropna = kwargs.pop("dropna", False)
    if isinstance(phi_bins, int):
        phi_range = (0, 2 * np.pi)
        if "phi_range" in "kwargs":
            phi_range = kwargs["phi_range"]
        elif "range" in "kwargs":
            phi_range = kwargs["range"][1]
        phi_range = list(phi_range) + [phi_bins + 1]
        phi_bins = np.linspace(*phi_range)

    if dropna:
        data = data[~np.isnan(data).any(axis=1)]
    bin_schemas = binnings.calculate_bins_nd(data, [radial_bins, phi_bins], *args, check_nan=not dropna, **kwargs)

    # Prepare remaining data
    weights = kwargs.pop("weights", None)
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(data, ndim=2,
                                                                  binnings=bin_schemas,
                                                                  weights=weights)
    return PolarHistogram(binnings=bin_schemas, frequencies=frequencies, errors2=errors2, missed=missed)


def spherical_histogram(data=None, radial_bins="human", theta_bins=16, phi_bins=16, transformed=False, *args, **kwargs):
    if not transformed:
        rdata, thetadata, phidata = SphericalHistogram.transform(data.T)
    else:
        rdata, thetadata, phidata = data.T
    data = np.concatenate([rdata[:, np.newaxis], thetadata[:, np.newaxis], phidata[:, np.newaxis]], axis=1)

    dropna = kwargs.pop("dropna", False)
    if dropna:
        data = data[~np.isnan(data).any(axis=1)]

    bin_schemas = binnings.calculate_bins_nd(data, [radial_bins, theta_bins, phi_bins], *args,
                                             check_nan=not dropna, **kwargs)
    weights = kwargs.pop("weights", None)
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(data, ndim=3,
                                                                  binnings=bin_schemas,
                                                                  weights=weights)
    return SphericalHistogram(binnings=bin_schemas, frequencies=frequencies, errors2=errors2, missed=missed)
