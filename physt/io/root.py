import os

import uproot


def write_root(histogram, hfile, name):
    """Write histogram to an open ROOT file.

    :param hfile: updateable uproot file object
    """
    hfile[name] = histogram


def save_root(histogram, path, name=None):
    """Write histogram to a (new) ROOT file.
    
    :param path: path for the output file.
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