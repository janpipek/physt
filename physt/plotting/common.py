"""
Functions that are shared by several (all) backends.

"""
from __future__ import absolute_import
import numpy as np


def get_data(h, density=False, cumulative=False, flatten=False):
    if density:
        if cumulative:
            data = (h / h.total).cumulative_frequencies
        else:
            data = h.densities
    else:
        if cumulative:
            data = h.cumulative_frequencies
        else:
            data = h.frequencies

    if flatten:
        data = data.flatten()
    return data


def get_err_data(h, density=False, cumulative=False, flatten=False):
    if cumulative:
        raise RuntimeError("Error bars not supported for cumulative plots.")
    if density:
        data = h.errors / h.bin_sizes
    else:
        data = h.errors
    if flatten:
        data = data.flatten()
    return data
