"""
physt
=====

P(i/y)thon h(i/y)stograms. Inspired (and based on) numpy.histogram,
but designed for humans(TM) on steroids(TM).

(C) Jan Pipek, 2016-2022, MIT licence
See https://github.com/janpipek/physt
"""
__all__ = [
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

from . import binnings, compat  # noqa: F401
from ._facade import (
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
from .version import __author__, __author_email__, __url__, __version__
