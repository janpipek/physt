from __future__ import annotations

from typing import TYPE_CHECKING

from physt._util import find_subclass
from physt.io.version import require_compatible_version
from physt.types import HistogramBase, HistogramCollection

if TYPE_CHECKING:
    from typing import Type, Union


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
