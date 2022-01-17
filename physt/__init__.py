"""
physt
=====

P(i/y)thon h(i/y)stograms. Inspired (and based on) numpy.histogram,
but designed for humans(TM) on steroids(TM).

(C) Jan Pipek, 2016-2021, MIT licence
See https://github.com/janpipek/physt
"""
__all__ = [
    "binnings",
    "compat",
    "special_histograms",
    "azimuthal",
    "collection",
    "cylindrical_surface",
    "cylindrical",
    "h",
    "h1",
    "h2",
    "h3",
    "polar",
    "radial",
    "spherical_surface",
    "spherical",
    "__author__",
    "__author_email__",
    "__url__",
    "__version__",
]

from facade import (
    azimuthal,
    collection,
    cylindrical,
    cylindrical_surface,
    h,
    h1,
    h2,
    h3,
    polar,
    radial,
    spherical,
    spherical_surface,
)

from . import binnings, compat  # noqa: F401
from .version import __author__, __author_email__, __url__, __version__
