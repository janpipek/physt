"""
physt
=====

P(i/y)thon h(i/y)stograms. Inspired (and based on) numpy.histogram, but designed for humans(TM) on steroids(TM).

The goal is to unify different concepts of histograms as occurring in numpy, pandas, matplotlib, ROOT, etc.
and to create one representation that is easily manipulated with from the data point of view and at the same
time provides nice integration into IPython notebook and various plotting options.

In short, whatever you want to do with histograms, physt aims to be on your side.

P.S. I am looking for anyone interested in using / developing physt. You can contribute by reporting errors,
implementing missing features and suggest new one.
"""

import itertools
import os
from setuptools import setup, find_packages


def read_info():
    """Single source of version number and other info.

    Inspiration:
    - https://packaging.python.org/guides/single-sourcing-package-version/
    - https://github.com/psf/requests/blob/master/setup.py
    """
    scope = {}
    version_file = os.path.join(THIS_DIR, "physt", "version.py")
    with open(version_file, "r") as f:
        exec(f.read(), scope)  # pylint: disable=exec-used
    return scope

THIS_DIR = os.path.dirname(__file__)
INFO = read_info()

options = dict(
    name="physt",
    version=INFO["__version__"],
    packages=find_packages(),
    license="MIT",
    description="P(i/y)thon h(i/y)stograms.",
    long_description=__doc__.strip(),
    author=INFO["__author__"],
    author_email=INFO["__author_email__"],
    url=INFO["__url__"],
    package_data={"physt": ["examples/*.csv"]},
    install_requires=["numpy>=1.17", "packaging"],
    python_requires="~=3.6",
    extras_require={
        "all": [
            "dask[array]",
            "toolz",
            "pandas",
            "matplotlib",
            "folium",
            "vega3",
            "xarray",
            "protobuf",
            "uproot3",  # TODO: Update to uproot4
            "asciiplotlib",
            "xtermcolor",
        ]
    },
    entry_points={"console_scripts": []},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

extras = options["extras_require"]
extras["full"] = list(set(itertools.chain.from_iterable(extras.values())))
setup(**options)
