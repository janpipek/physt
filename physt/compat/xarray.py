"""Xarray integration.

- conversion between 1D histograms and xarray Datasets.

This is experimental and may change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xarray import DataArray, Dataset

from physt.types import Histogram1D

if TYPE_CHECKING:
    from typing import Any, Dict, Type


def _h1_to_xarray(h1: Histogram1D) -> Dataset:
    """Convert histogram to an xarray Datates representation.

    See also: Histogram1D.from_xarray (inverse operation)
    """

    # TODO: Rethink how the dimensions / variables are used

    data_vars: Dict[str, Any] = {
        "frequencies": DataArray(h1.frequencies, dims="bin"),
        "errors2": DataArray(h1.errors2, dims="bin"),
        "bins": DataArray(h1.bins, dims=("bin", "x01")),
    }
    coords: Dict[str, Any] = {}
    attrs: Dict[str, Any] = {
        "underflow": h1.underflow,
        "overflow": h1.overflow,
        "inner_missed": h1.inner_missed,
        "keep_missed": h1.keep_missed,
    }
    attrs.update(h1.meta_data)
    # TODO: Add stats
    return Dataset(data_vars, coords, attrs)


def _h1_from_xarray(cls: Type[Histogram1D], arr: Dataset) -> Histogram1D:
    """Convert form xarray.Dataset

    Parameters
    ----------
    arr: The data in xarray representation
    """
    kwargs = {
        "frequencies": arr["frequencies"],
        "binning": arr["bins"],
        "errors2": arr["errors2"],
    }
    kwargs.update(arr.attrs)  # type: ignore
    # TODO: Add stats
    return cls(**kwargs)  # type: ignore


setattr(Histogram1D, "to_xarray", _h1_to_xarray)
setattr(Histogram1D, "from_xarray", classmethod(_h1_from_xarray))


# TODO: Implement multi-dimensional histograms
