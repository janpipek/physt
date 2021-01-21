from typing import Union, Type

from physt.io.version import require_compatible_version
from physt.util import find_subclass
from physt.histogram_base import HistogramBase
from physt.histogram_collection import HistogramCollection


def create_from_dict(
    data: dict, format_name: str, check_version: bool = True
) -> Union[HistogramBase, HistogramCollection]:
    """Once dict from source data is created, turn this into histogram.

    Parameters
    ----------
    data : Parsed JSON-like tree.

    Returns
    -------
    histogram :  A histogram (of any dimensionality)
    """
    # Version
    if check_version:
        compatible_version = data["physt_compatible"]
        require_compatible_version(compatible_version, format_name)

    # Construction
    histogram_type = data["histogram_type"]
    if histogram_type == "histogram_collection":
        return HistogramCollection.from_dict(data)
    klass: Type[HistogramBase] = find_subclass(HistogramBase, histogram_type)
    return klass.from_dict(data)
