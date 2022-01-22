"""Histograms types and function for various external libraries."""

__all__ = []

try:
    from . import pandas

    __all__.append("pandas")
except ImportError:
    pass


try:
    from . import dask

    __all__.append("dask")
except ImportError:
    pass


try:
    from . import geant4

    __all__.append("geant4")
except ImportError:
    pass


# TODO: Make xarray a compat too.
