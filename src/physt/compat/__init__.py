"""Histograms types and function for various external libraries."""

__all__ = []

try:
    from . import pandas  # noqa: F401

    __all__.append("pandas")
except ImportError:
    pass


try:
    from . import dask  # noqa: F401

    __all__.append("dask")
except ImportError:
    pass


try:
    from . import geant4  # noqa: F401

    __all__.append("geant4")
except ImportError:
    pass


try:
    from . import xarray  # noqa: F401

    __all__.append("xarray")
except ImportError:
    pass
