from collections import OrderedDict
from . import __version__
import json


def save_json(histogram, path=None):
    data = histogram.to_dict()
    data["physt-version"] = __version__
    data["physt-compatible"] = "0.3.20"

    text = json.dumps(data)
    if path:
        with open(path, "w", encoding="ascii") as f:
            f.write(text)
    return text


def load_json(path):
    pass
