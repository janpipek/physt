#!/usr/bin/env python
"""
physt
=====

P(i/y)thon h(i/y)stograms. Inspired (and based on) numpy.histogram, but designed for humans(TM) on steroids(TM).

The goal is to unify different concepts of histograms as occurring in numpy, pandas, matplotlib, ROOT, etc.
and to create one representation that is easily manipulated with from the data point of view and at the same
time provides nice integration into IPython notebook and various plotting options.

In short, whatever you want to do with histograms, physt aims to be on your side.

P.S. I am looking for anyone interested in using / developing physt. You can contribute by reporting errors, implementing missing features and suggest new one.
"""

import itertools
from setuptools import setup, find_packages
# from physt.version import __version__

VERSION = "0.4"

options = dict(
    name="physt",
    version=VERSION,
    packages=find_packages(),
    license="MIT",
    description="P(i/y)thon h(i/y)stograms.",
    long_description=__doc__.strip(),
    author="Jan Pipek",
    author_email="jan.pipek@gmail.com",
    url="https://github.com/janpipek/physt",
    package_data={"physt" : ["examples/*.csv"]},
    install_requires = ["numpy"],
    python_requires = "~=3.5",
    extras_require = {
        # 'all' : ['dask', 'matplotlib', 'bokeh', 'folium']
    },
    entry_points = {
        'console_scripts' : [
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)

extras = options['extras_require']
extras['all'] = list(set(itertools.chain.from_iterable(extras.values())))
setup(**options)
