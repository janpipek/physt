"""Histograms types and function for various external libraries."""
from contextlib import suppress

__all__ = []

with suppress(ImportError):
    from . import pandas  # noqa: F401

    __all__.append("pandas")

with suppress(ImportError):
    from . import dask  # noqa: F401

    __all__.append("dask")

with suppress(ImportError):
    from . import geant4  # noqa: F401

    __all__.append("geant4")

with suppress(ImportError):
    from . import xarray  # noqa: F401

    __all__.append("xarray")


with suppress(ImportError):
    from . import polars  # noqa: F401

    __all__.append("polars")
