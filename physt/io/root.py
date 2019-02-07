import os
from typing import Optional

import uproot

from physt.histogram_base import HistogramBase


def write_root(histogram: HistogramBase, hfile: uproot.write.TFile.TFileUpdate, name: str):
    """Write histogram to an open ROOT file.

    :param hfile: updateable uproot file object
    """
    hfile[name] = histogram


def save_root(histogram: HistogramBase, path: str, name: Optional[str] = None):
    """Write histogram to a (new) ROOT file.
    
    :param path: path for the output file (perhaps should not exist?)
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
