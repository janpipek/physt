"""Input and output for histograms.

JSON format is included by default.
Other formats are/will be available as modules.

Note: When implementing, try to work with a JSON-like
  tree and reuse `create_from_dict` and `HistogramBase.to_dict`.
"""
from packaging.version import Version
from pkg_resources import parse_version
from typing import Union

from physt import __version__
from physt.util import find_subclass
from physt.histogram_base import HistogramBase
from physt.histogram_collection import HistogramCollection

CURRENT_VERSION = __version__


class VersionError(Exception):
    pass


def create_from_dict(data: dict, format_name: str, check_version: bool = True) -> Union[HistogramBase, HistogramCollection]:
    """Once dict from source data is created, turn this into histogram.
    
    Parameters
    ----------
    data : dict
        Parsed JSON-like tree.

    Returns
    -------
    histogram : HistogramBase
        A histogram (of any dimensionality)
    """
    # Version
    if check_version:
        compatible_version = data["physt_compatible"]
        require_compatible_version(compatible_version, format_name)

    # Construction
    histogram_type = data["histogram_type"]
    if histogram_type == "histogram_collection":
        klass = HistogramCollection
    else:
        klass = find_subclass(HistogramBase, histogram_type)
    return klass.from_dict(data)


def require_compatible_version(compatible_version, word="File"):
    """Check that compatible version of input data is not too new."""
    if isinstance(compatible_version, str):
        compatible_version = parse_version(compatible_version)
    elif not isinstance(compatible_version, Version):
        raise ValueError("Type of `compatible_version` not understood.")
    
    current_version = parse_version(CURRENT_VERSION)
    if current_version < compatible_version:
        raise VersionError("{0} written for version >= {1}, this is {2}.".format(
            word, str(compatible_version), CURRENT_VERSION
        ))


from .json import save_json, load_json, parse_json
