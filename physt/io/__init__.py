"""Input and output for histograms.

JSON format is included by default.
Other formats are/will be available as modules.

Note: When implementing, try to work with a JSON-like
  tree and reuse `create_from_dict` and `HistogramBase.to_dict`.
"""

from .json import save_json, load_json, parse_json
from .util import create_from_dict

__all__ = ["save_json", "load_json", "parse_json", "create_from_dict"]

# Optional ROOT support
try:
    from .root import write_root, save_root

    __all__.extend(["write_root", "save_root"])
except ImportError:
    pass
