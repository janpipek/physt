import json

from . import CURRENT_VERSION, create_from_dict
from ..histogram_base import HistogramBase
from ..util import find_subclass

COMPATIBLE_VERSION = "0.3.20"


def save_json(histogram, path=None, **kwargs):
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


def load_json(path=None):
    """Load histogram from a JSON file.

    Parameters
    ----------
    path : str
        Path to the histogram file.

    Returns
    -------
    hist : HistogramBase
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        return parse_json(text)


def parse_json(text):
    """Create histogram from a JSON string.

    Parameters
    ----------
    text : str
        UTF-8 encoded JSON
    """
    data = json.loads(text, encoding="utf-8")
    return create_from_dict(data, format_name="JSON")