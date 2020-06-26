"""Transformed histograms.

These histograms use a transformation from input values to bins
in a different coordinate system.

There are three basic classes:

* PolarHistogram
* CylindricalHistogram
* SphericalHistogram

Apart from these, there are their projections into lower dimensions.

And of course, it is possible to re-use the general transforming functionality
by adding `TransformedHistogramMixin` among the custom histogram
class superclasses.
"""
import abc
from functools import reduce
from typing import Optional, Tuple

import numpy as np

from .histogram_nd import HistogramND
from .histogram1d import Histogram1D
from .typing_aliases import RangeTuple
from . import binnings, histogram_nd


FULL_PHI_RANGE: RangeTuple = (0, 2 * np.pi)
FULL_THETA_RANGE: RangeTuple = (0, np.pi)


class TransformedHistogramMixin(abc.ABC):
    """Histogram with non-cartesian (or otherwise transformed) axes.

    This is a mixin, providing transform-aware find_bin, fill and fill_n.

    When implementing, you are required to provide tbe following:
    - `transform` method to convert rectangular (suggested to make it classmethod)
    - `bin_sizes` property

    In certain cases, you may want to have default axis names + projections.
    Look at PolarHistogram / SphericalHistogram / CylindricalHistogram as
    an example.
    """

    @classmethod
    # @abc.abstractmethod
    def transform(cls, value):
        """Convert cartesian (general) coordinates into internal ones.

        Parameters
        ----------
        value : array_like
            This method should accept both scalars and numpy arrays.
            If multiple values are to be transformed, it should of
            (nvalues, ndim) shape.

        Returns
        -------
        float or array_like
        """
        ...

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
    @abc.abstractmethod
    def bin_sizes(self):
        ...

    def fill(self, value, weight=1, transformed=False):
        return HistogramND.fill(
            self, value=value, weight=weight, transformed=transformed
        )

    def fill_n(self, values, weights=None, dropna=True, transformed=False):
        if not transformed:
            values = self.transform(values)
        HistogramND.fill_n(self, values=values, weights=weights, dropna=dropna)

    _projection_class_map = {}

    def projection(self, *axes, **kwargs):
        """Projection to lower-dimensional histogram.
        
        The inheriting class should implement the _projection_class_map
        class attribute to suggest class for the projection. If the 
        arguments don't match any of the map keys, HistogramND is used.        
        """
        axes, _ = self._get_projection_axes(*axes)
        axes = tuple(sorted(axes))
        if axes in self._projection_class_map:
            klass = self._projection_class_map[axes]
            return HistogramND.projection(self, *axes, type=klass, **kwargs)
        else:
            return HistogramND.projection(self, *axes, **kwargs)


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


class PolarHistogram(TransformedHistogramMixin, HistogramND):
    """2D histogram in polar coordinates.

    This is a special case of a 2D histogram with transformed coordinates:
    - r as radius in the (0, +inf) range
    - phi as azimuthal angle in the (0, 2*pi) range

    """

    def __init__(self, binnings, frequencies=None, **kwargs):
        if not "axis_names" in kwargs:
            kwargs["axis_names"] = ("r", "phi")
        if "dim" in kwargs:
            kwargs.pop("dim")
        super(PolarHistogram, self).__init__(
            2, binnings=binnings, frequencies=frequencies, **kwargs
        )

    @property
    def bin_sizes(self):
        sizes = 0.5 * (
            self.get_bin_right_edges(0) ** 2 - self.get_bin_left_edges(0) ** 2
        )
        sizes = np.outer(sizes, self.get_bin_widths(1))
        return sizes

    @classmethod
    def transform(cls, value):
        value = np.asarray(value, dtype=np.float64)
        assert value.shape[-1] == 2
        result = np.empty_like(value)
        result[..., 0] = np.hypot(value[..., 1], value[..., 0])
        result[..., 1] = np.arctan2(value[..., 1], value[..., 0]) % (2 * np.pi)
        return result

    _projection_class_map = {(0,): RadialHistogram, (1,): AzimuthalHistogram}


class SphericalSurfaceHistogram(TransformedHistogramMixin, HistogramND):
    """2D histogram in spherical coordinates.

    This is a special case of a 2D histogram with transformed coordinates:
    - theta as angle between z axis and the vector, in the (0, 2*pi) range
    - phi as azimuthal angle  (in the xy projection) in the (0, 2*pi) range
    """

    @property
    def bin_sizes(self):
        sizes1 = np.cos(self.get_bin_left_edges(0)) - np.cos(
            self.get_bin_right_edges(0)
        )
        sizes2 = self.get_bin_widths(1)
        return reduce(np.multiply, np.ix_(sizes1, sizes2))

    def __init__(self, binnings, frequencies=None, radius=1, **kwargs):
        if "axis_names" not in kwargs:
            kwargs["axis_names"] = ("theta", "phi")
        if "dim" in kwargs:
            kwargs.pop("dim")
        super(SphericalSurfaceHistogram, self).__init__(
            2, binnings=binnings, frequencies=frequencies, **kwargs
        )
        self.radius = radius

    @property
    def radius(self):
        """Radius of the surface.

        Useful for calculating densities.
        """
        return self._meta_data.get("radius", 1)

    @classmethod
    def transform(cls, value):
        if not len(value.shape) == 2 or value.shape[1] != 3:
            raise ValueError(
                f"{cls.__name__} can project only tables with shape (N, 3), {value.shape} given."
            )
        result = np.ndarray((value.shape[0], 2), dtype=np.float64)
        x, y, z = value.T
        xy = np.hypot(x, y)
        result[..., 0] = np.arctan2(xy, z) % (2 * np.pi)
        result[..., 1] = np.arctan2(y, x) % (2 * np.pi)
        return result

    @radius.setter
    def radius(self, value):
        self._meta_data["radius"] = value


class SphericalHistogram(TransformedHistogramMixin, HistogramND):
    """3D histogram in spherical coordinates.

    This is a special case of a 3D histogram with transformed coordinates:
    - r as radius in the (0, +inf) range
    - theta as angle between z axis and the vector, in the (0, 2*pi) range
    - phi as azimuthal angle  (in the xy projection) in the (0, 2*pi) range
    """

    def __init__(self, binnings, frequencies=None, **kwargs):
        if "axis_names" not in kwargs:
            kwargs["axis_names"] = ("r", "theta", "phi")
        kwargs.pop("dim", False)
        super(SphericalHistogram, self).__init__(
            3, binnings=binnings, frequencies=frequencies, **kwargs
        )

    @classmethod
    def transform(cls, value):
        value = np.asarray(value, dtype=np.float64)
        result = np.empty_like(value)
        x, y, z = value.T
        xy = np.hypot(x, y)
        result[..., 0] = np.hypot(xy, z)
        result[..., 1] = np.arctan2(xy, z) % (2 * np.pi)
        result[..., 2] = np.arctan2(y, x) % (2 * np.pi)
        return result

    @property
    def bin_sizes(self):
        sizes1 = (
            self.get_bin_right_edges(0) ** 3 - self.get_bin_left_edges(0) ** 3
        ) / 3
        sizes2 = np.cos(self.get_bin_left_edges(1)) - np.cos(
            self.get_bin_right_edges(1)
        )
        sizes3 = self.get_bin_widths(2)
        # Hopefully correct
        return reduce(np.multiply, np.ix_(sizes1, sizes2, sizes3))
        # return np.outer(sizes, sizes2, self.get_bin_widths(2))    # Correct

    _projection_class_map = {(1, 2): SphericalSurfaceHistogram, (0,): RadialHistogram}


class CylindricalSurfaceHistogram(TransformedHistogramMixin, HistogramND):
    """2D histogram in coordinates on cylinder surface.

    This is a special case of a 2D histogram with transformed coordinates:
    - phi as azimuthal angle  (in the xy projection) in the (0, 2*pi) range
    - z as the last direction without modification, in (-inf, +inf) range

    Attributes
    ----------
    radius: float
        The radius of the surface. Useful for plotting
    """

    def __init__(self, binnings, frequencies=None, radius=1, **kwargs):
        if not "axis_names" in kwargs:
            kwargs["axis_names"] = ("phi", "z")
        if "dim" in kwargs:
            kwargs.pop("dim")
        super(CylindricalSurfaceHistogram, self).__init__(
            2, binnings=binnings, frequencies=frequencies, **kwargs
        )
        self.radius = radius

    @property
    def radius(self):
        """Radius of the cylindrical surface.

        Useful for calculating densities.

        Returns
        -------
        float
        """
        return self._meta_data.get("radius", 1)

    @property
    def bin_sizes(self):
        sizes1 = self.get_bin_widths(0)
        sizes2 = self.get_bin_widths(1)
        return reduce(np.multiply, np.ix_(sizes1, sizes2))

    @radius.setter
    def radius(self, value):
        self._meta_data["radius"] = float(value)

    _projection_class_map = {(0,): AzimuthalHistogram}


class CylindricalHistogram(TransformedHistogramMixin, HistogramND):
    """3D histogram in cylindrical coordinates.

    This is a special case of a 3D histogram with transformed coordinates:
    - r as radius projection to xy plane in the (0, +inf) range
    - phi as azimuthal angle  (in the xy projection) in the (0, 2*pi) range
    - z as the last direction without modification, in (-inf, +inf) range
    """

    def __init__(self, binnings, frequencies=None, **kwargs):
        if not "axis_names" in kwargs:
            kwargs["axis_names"] = ("rho", "phi", "z")
        kwargs.pop("dim", False)
        super(CylindricalHistogram, self).__init__(
            3, binnings=binnings, frequencies=frequencies, **kwargs
        )

    @classmethod
    def transform(cls, value):
        value = np.asarray(value, dtype=np.float64)
        result = np.empty_like(value)
        x, y, z = value.T
        result[..., 0] = np.hypot(x, y)  # tho
        result[..., 1] = np.arctan2(y, x) % (2 * np.pi)  # phi
        result[..., 2] = z
        return result

    @property
    def bin_sizes(self):
        sizes1 = 0.5 * (
            self.get_bin_right_edges(0) ** 2 - self.get_bin_left_edges(0) ** 2
        )
        sizes2 = self.get_bin_widths(1)
        sizes3 = self.get_bin_widths(2)
        return reduce(np.multiply, np.ix_(sizes1, sizes2, sizes3))

    _projection_class_map = {
        (0,): RadialHistogram,
        (1,): AzimuthalHistogram,
        (0, 1): PolarHistogram,
        (1, 2): CylindricalSurfaceHistogram,
    }

    def projection(self, *args, **kwargs):
        result = TransformedHistogramMixin.projection(self, *args, **kwargs)
        if isinstance(result, CylindricalSurfaceHistogram):
            result.radius = self.get_bin_right_edges(0)[-1]
        return result


def _prepare_data(data, transformed: bool, klass, *, dropna: bool = False):
    """Transform data for binning.

    Returns
    -------
    np.ndarray
    """
    # TODO: Maybe include in the class itself?
    data = np.asarray(data)
    if not transformed:
        data = klass.transform(data)
    if dropna:
        data = data[~np.isnan(data).any(axis=1)]
    return data


def polar_histogram(
    xdata,
    ydata,
    *,
    radial_bins="numpy",
    radial_range: Optional[RangeTuple] = None,
    phi_bins=16,
    phi_range: RangeTuple = (0, 2 * np.pi),
    dropna: bool = False,
    transformed: bool = False,
    **kwargs
):
    """Facade construction function for the PolarHistogram."""
    if "range" in kwargs:
        raise ValueError("Please, use `radial_range` and `phi_range` arguments instead of `range`")

    data = np.concatenate([xdata[:, np.newaxis], ydata[:, np.newaxis]], axis=1)
    data = _prepare_data(
        data, transformed=transformed, klass=PolarHistogram, dropna=dropna
    )

    if isinstance(phi_bins, int):
        phi_bins = np.linspace(*phi_range, phi_bins + 1)

    bin_schemas = binnings.calculate_bins_nd(
        data, [radial_bins, phi_bins], range=[radial_range, None], check_nan=not dropna, **kwargs
    )
    weights = kwargs.pop("weights", None)
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(
        data, ndim=2, binnings=bin_schemas, weights=weights
    )
    return PolarHistogram(
        binnings=bin_schemas, frequencies=frequencies, errors2=errors2, missed=missed
    )


def spherical_histogram(
    data=None,
    *,
    radial_bins="numpy",
    theta_bins=16,
    phi_bins=16,
    dropna: bool = True,
    transformed: bool = False,
    theta_range: RangeTuple = (0, np.pi),
    phi_range: RangeTuple = (0, 2 * np.pi),
    radial_range: Optional[RangeTuple] = None,
    **kwargs,
):
    """Facade construction function for the SphericalHistogram."""
    if "range" in kwargs:
        raise ValueError("Please, use `radial_range`, `theta_range` and `phi_range` arguments instead of `range`")

    data = _prepare_data(
        data, transformed=transformed, klass=SphericalHistogram, dropna=dropna
    )

    if isinstance(theta_bins, int):
        theta_bins = np.linspace(*theta_range, theta_bins + 1)

    if isinstance(phi_bins, int):
        phi_bins = np.linspace(*phi_range, phi_bins + 1)

    try:
        bin_schemas = binnings.calculate_bins_nd(
            data, [radial_bins, theta_bins, phi_bins], range=[radial_range, None, None], check_nan=not dropna, **kwargs
        )
    except RuntimeError as err:
        if "Bins not in rising order" in str(err):
            import warnings

            if np.isclose(data[:, 0].min(), data[:, 0].max()):
                raise ValueError(
                    f"All radii seem to be the same: {data[:,0].min():,.4f}. "
                    "Perhaps you wanted to use `spherical_surface_histogram` instead or set radius bins explicitly?"
                )
        raise

    weights = kwargs.pop("weights", None)
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(
        data, ndim=3, binnings=bin_schemas, weights=weights
    )

    return SphericalHistogram(
        binnings=bin_schemas, frequencies=frequencies, errors2=errors2, missed=missed
    )


def spherical_surface_histogram(
    data=None,
    *,
    theta_bins=16,
    phi_bins=16,
    transformed: bool = False,
    radius=None,
    dropna: bool = False,
    weights=None,
    theta_range: RangeTuple = FULL_THETA_RANGE,
    phi_range: RangeTuple = FULL_PHI_RANGE,
    **kwargs,
):
    """Facade construction function for the SphericalSurfaceHistogram."""
    transformed_data = _prepare_data(
        data, transformed=transformed, klass=SphericalSurfaceHistogram, dropna=dropna
    )

    if "range" in kwargs:
        raise ValueError("Please, use `theta_range` and `phi_range` arguments instead of `range`")

    if transformed_data is not None:
        if not transformed and radius is None:
            radius = np.hypot(np.hypot(data[:, 0], data[:, 1]), data[:, 2])

    if radius is None:
        radius = 1

    if isinstance(theta_bins, int):
        theta_bins = np.linspace(*theta_range, theta_bins + 1)

    if isinstance(phi_bins, int):
        phi_bins = np.linspace(*phi_range, phi_bins + 1)

    bin_schemas = binnings.calculate_bins_nd(
        transformed_data, [theta_bins, phi_bins], check_nan=not dropna, **kwargs
    )
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(
        transformed_data, ndim=2, binnings=bin_schemas, weights=weights
    )
    return SphericalSurfaceHistogram(
        binnings=bin_schemas,
        frequencies=frequencies,
        errors2=errors2,
        missed=missed,
        radius=radius,
    )


def cylindrical_histogram(
    data=None,
    *,
    rho_bins="numpy",
    phi_bins=16,
    z_bins="numpy",
    transformed: bool = False,
    dropna: bool = True,
    rho_range: Optional[RangeTuple] = None,
    phi_range: RangeTuple = FULL_PHI_RANGE,
    z_range=None,
    **kwargs,
):
    """Facade construction function for the CylindricalHistogram."""
    if "range" in kwargs:
        raise ValueError("Please, use `rho_range`, `phi_range` and `z_range` arguments instead of `range`")

    data = _prepare_data(
        data, transformed=transformed, klass=CylindricalHistogram, dropna=dropna
    )

    if isinstance(phi_bins, int):
        phi_bins = np.linspace(*phi_range, phi_bins + 1)

    bin_schemas = binnings.calculate_bins_nd(
        data, [rho_bins, phi_bins, z_bins], range=[rho_range, None, z_range], check_nan=not dropna, **kwargs
    )
    weights = kwargs.pop("weights", None)
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(
        data, ndim=3, binnings=bin_schemas, weights=weights
    )
    return CylindricalHistogram(
        binnings=bin_schemas, frequencies=frequencies, errors2=errors2, missed=missed
    )


def cylindrical_surface_histogram(
    data=None,
    *,
    phi_bins=16,
    z_bins="numpy",
    transformed: bool = False,
    radius=None,
    dropna: bool = False,
    weights=None,
    phi_range: RangeTuple = FULL_PHI_RANGE,
    z_range: Optional[RangeTuple] = None,
    **kwargs,
):
    """Facade construction function for the CylindricalSurfaceHistogram."""
    if "range" in kwargs:
        raise ValueError("Please, use `phi_range` and `z_range` arguments instead of `range`")

    transformed_data = _prepare_data(
        data, transformed=transformed, klass=CylindricalHistogram, dropna=dropna
    )

    if transformed_data is not None:
        if not transformed and radius is None:
            radius = np.hypot(data[:, 0], data[:, 1])
    if radius is None:
        radius = 1

    if isinstance(phi_bins, int):
        phi_bins = np.linspace(*phi_range, phi_bins + 1)

    bin_schemas = binnings.calculate_bins_nd(
        transformed_data, [phi_bins, z_bins], range=[None, z_range], check_nan=not dropna, **kwargs
    )
    frequencies, errors2, missed = histogram_nd.calculate_frequencies(
        data, ndim=3, binnings=bin_schemas, weights=weights
    )
    return CylindricalSurfaceHistogram(
        binnings=bin_schemas,
        frequencies=frequencies,
        errors2=errors2,
        radius=radius,
        missed=missed,
    )
