import json
from typing import Optional

from physt.io import CURRENT_VERSION, create_from_dict
from physt.histogram_base import HistogramBase
from physt.util import find_subclass

COMPATIBLE_VERSION = "0.3.20"


def save_json(histogram: HistogramBase, path: Optional[str] = None, **kwargs) -> str:
    """Save histogram to JSON format.

    Parameters
    ----------
    histogram : HistogramBase
        Any histogram
    path : str
        If set, also writes to the path.

    Returns
    -------
    json : str
        The JSON representation of the histogram
    """
    # TODO: Implement multiple histograms in one file?
    data = histogram.to_dict()
    data["physt_version"] = CURRENT_VERSION
    data["physt_compatible"] = COMPATIBLE_VERSION

    text = json.dumps(data, **kwargs)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def load_json(path: str, encoding: str = "utf-8") -> HistogramBase:
    """Load histogram from a JSON file."""
    with open(path, "r", encoding=encoding) as f:
        text = f.read()
        return parse_json(text)


def parse_json(text: str, encoding: str = "utf-8") -> HistogramBase:
    """Create histogram from a JSON string."""
    data = json.loads(text, encoding=encoding)
    return create_from_dict(data, format_name="JSON")
