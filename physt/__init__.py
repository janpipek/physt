"""
physt
=====

P(i/y)thon h(i/y)stograms. Inspired (and based on) numpy.histogram,
but designed for humans(TM) on steroids(TM).

(C) Jan Pipek, 2016-2021, MIT licence
See https://github.com/janpipek/physt
"""

from . import binnings
from . import special_histograms
from . import compat
from .facade import *
from .version import __version__, __author__, __author_email__, __url__
