"""Support for Geant4 histograms saved in CSV format."""

from __future__ import absolute_import
import codecs

import numpy as np

from ..histogram1d import Histogram1D
from ..histogram_nd import Histogram2D
from ..binnings import fixed_width_binning


def load_csv(path):
    """Loads a histogram as output from Geant4 analysis tools in CSV format.

    Parameters
    ----------
    path: str
        Path to the CSV file

    Returns
    -------
    physt.histogram1d.Histogram1D or physt.histogram_nd.Histogram2D
    """
    meta = []
    data = []
    with codecs.open(path, encoding="ASCII") as in_file:
        for line in in_file:
            if line.startswith("#"):
                key, value = line[1:].strip().split(" ", 1)
                meta.append((key, value))   # TODO: There are duplicit entries :-()
            else:
                try:
                    data.append([float(frag) for frag in line.split(",")])
                except:
                    pass
    data = np.asarray(data)
    ndim = int(_get(meta, "dimension"))
    if ndim == 1:
        return _create_h1(data, meta)
    elif ndim == 2:
        return _create_h2(data, meta)


def _get(pseudodict, key, single=True):
    """Helper method for getting values from "multi-dict"s"""
    matches = [item[1] for item in pseudodict if item[0] == key]
    if single:
        return matches[0]
    else:
        return matches


def _create_h1(data, meta):
    _, bin_count, min_, max_ = _get(meta, "axis").split()
    bin_count = int(bin_count)
    min_ = float(min_)
    max_ = float(max_)
    binning = fixed_width_binning(None, bin_width=(max_ - min_) / bin_count, range=(min_, max_))
    hist = Histogram1D(binning, name=_get(meta, "title"))
    hist._frequencies = data[1:-1, 1]
    hist._errors2 = data[1:-1, 2]
    hist.underflow = data[0, 1]
    hist.overflow = data[-1, 1]
    hist._stats = {
        "sum" : data[1:-1, 3].sum(),
        "sum2" : data[1:-1, 4].sum()
    }
    return hist


def _create_h2(data, meta):
    binnings = []
    axes = _get(meta, "axis", False)
    for axis in axes:
        _, bin_count, min_, max_ = axis.split()
        bin_count = int(bin_count)
        min_ = float(min_)
        max_ = float(max_)
        binning = fixed_width_binning(None, bin_width=(max_ - min_) / bin_count, range=(min_, max_))
        binnings.append(binning)

    hist = Histogram2D(binnings, name=_get(meta, "title"))

    # TODO: Are the shapes in correct order?
    frequencies = data[:, 1].reshape([b + 2 for b in hist.shape])
    hist._frequencies = frequencies[1:-1, 1:-1]

    errors2 = data[:, 2].reshape([b + 2 for b in hist.shape])
    hist._errors = errors2[1:-1, 1:-1]

    return hist
