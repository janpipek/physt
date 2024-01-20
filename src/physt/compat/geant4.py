"""Support for Geant4 histograms saved in CSV format.

See https://geant4.web.cern.ch/ for the project pages.
"""
import codecs
from typing import Union

import numpy as np

from physt.binnings import fixed_width_binning
from physt.statistics import Statistics
from physt.types import Histogram1D, Histogram2D


def load_csv(path: str) -> Union[Histogram1D, Histogram2D]:
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
    data_raw = []
    with codecs.open(path, encoding="ASCII") as in_file:
        for line in in_file:
            if line.startswith("#"):
                key, value = line[1:].strip().split(" ", 1)
                meta.append((key, value))  # TODO: There are duplicit entries :-()
            else:
                try:  # noqa: FURB107
                    data_raw.append([float(frag) for frag in line.split(",")])
                except:  # noqa: E722  # TODO: Find out why
                    pass
    data = np.asarray(data_raw)
    ndim = int(_get(meta, "dimension"))
    if ndim == 1:
        return _create_h1(data, meta)
    if ndim == 2:
        return _create_h2(data, meta)
    raise ValueError(f"Cannot handle histograms with dimension > 2: {ndim}")


def _get(pseudodict, key, single=True):
    """Helper method for getting values from "multi-dict"s"""
    matches = [item[1] for item in pseudodict if item[0] == key]
    if single:
        return matches[0]
    return matches


def _create_h1(data, meta) -> Histogram1D:
    _, bin_count, min_, max_ = _get(meta, "axis").split()
    bin_count = int(bin_count)
    min_ = float(min_)
    max_ = float(max_)
    binning = fixed_width_binning(
        bin_width=(max_ - min_) / bin_count, range=(min_, max_)
    )
    stats = Statistics(sum=data[1:-1, 3].sum(), sum2=data[1:-1, 4].sum())

    hist = Histogram1D(
        binning,
        frequencies=data[1:-1, 1],
        errors2=data[1:-1, 2],
        name=_get(meta, "title"),
        underflow=data[0, 1],
        overflow=data[-1, 1],
        stats=stats
    )
    return hist


def _create_h2(data, meta) -> Histogram2D:
    binnings = []
    axes = _get(meta, "axis", False)
    for axis in axes:
        _, bin_count, min_, max_ = axis.split()
        bin_count = int(bin_count)
        min_ = float(min_)
        max_ = float(max_)
        binning = fixed_width_binning(
            bin_width=(max_ - min_) / bin_count, range=(min_, max_)
        )
        binnings.append(binning)

    shape = Histogram2D(binnings).shape

    # TODO: Are the shapes in correct order?
    frequencies = data[:, 1].reshape([b + 2 for b in shape])
    frequencies = frequencies[1:-1, 1:-1]

    errors2 = data[:, 2].reshape([b + 2 for b in shape])
    errors2 = errors2[1:-1, 1:-1]

    hist = Histogram2D(
        binnings=binnings,
        name=_get(meta, "title"),
        frequencies=frequencies,
        errors2=errors2,
    )

    return hist
