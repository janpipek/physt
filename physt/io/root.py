"""ROOT format I/O

See also
--------
- https://github.com/scikit-hep/uproot
- https://root.cern.ch
"""
import os
from typing import Optional

import uproot

from physt.histogram_base import HistogramBase


def write_root(histogram: HistogramBase, hfile: uproot.write.TFile.TFileUpdate, name: str):
    """Write histogram to an open ROOT file.

    Parameters
    ----------
    histogram : Any histogram
    hfile : Updateable uproot file object
    name : The name of the histogram inside the file
    """
    hfile[name] = histogram


def save_root(histogram: HistogramBase, path: str, name: Optional[str] = None):
    """Write histogram to a (new) ROOT file.

    Parameters
    ----------
    histogram : Any histogram
    path: path for the output file (perhaps should not exist?)
    name : The name of the histogram inside the file
    """
    if name is None:
        name = histogram.name
    if os.path.isfile(path):
        # TODO: Not supported currently
        hfile = uproot.write.TFile.TFileUpdate(path)
    else:
        hfile = uproot.write.TFile.TFileCreate(path)
    write_root(histogram, hfile, name)
    hfile.close()
