"""
physt
=====

P(i/y)thon h(i/y)stograms. Inspired (and based on) numpy.histogram,
but designed for humans(TM) on steroids(TM).

(C) Jan Pipek, 2016-2021, MIT licence
See https://github.com/janpipek/physt
"""

from . import binnings, compat, special_histograms
from .facade import *
from .version import __author__, __author_email__, __url__, __version__
