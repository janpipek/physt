"""JSON I/O."""

import json
from pathlib import Path
from typing import Union

from physt.io.util import create_from_dict
from physt.io.version import CURRENT_VERSION
from physt.types import HistogramBase, HistogramCollection

COMPATIBLE_VERSION = "0.3.20"
"""The oldest version of physt that should be able to read the stored histograms."""

COLLECTION_COMPATIBLE_VERSION = "0.4.5"
"""The oldest version of physt that should be able to read the stored histogram collections."""


def save_json(
    histogram: Union[HistogramBase, HistogramCollection],
    path: Union[str, Path, None] = None,
    **kwargs,
) -> str:
    """Save histogram to JSON format.

    Parameters
    ----------
    histogram : Any histogram
    path : If set, also writes to the path.

    Returns
    -------
    json : The JSON representation of the histogram
    """
    # TODO: Implement multiple histograms in one file?
    data = histogram.to_dict()

    data["physt_version"] = CURRENT_VERSION
    if isinstance(histogram, HistogramBase):
        data["physt_compatible"] = COMPATIBLE_VERSION
    elif isinstance(histogram, HistogramCollection):
        data["physt_compatible"] = COLLECTION_COMPATIBLE_VERSION
    else:
        raise TypeError(f"Cannot save unknown type: {type(histogram)}")

    text = json.dumps(data, **kwargs)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def load_json(
    path: Union[str, Path], encoding: str = "utf-8"
) -> Union[HistogramBase, HistogramCollection]:
    """Load histogram from a JSON file."""
    with open(path, "r", encoding=encoding) as f:
        text = f.read()
        return parse_json(text)


def parse_json(text: str) -> Union[HistogramBase, HistogramCollection]:
    """Create histogram from a JSON string."""
    data = json.loads(text)
    return create_from_dict(data, format_name="JSON")
