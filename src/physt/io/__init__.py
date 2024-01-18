# noqa: # noqa: F401
"""Input and output for histograms.

JSON format is included by default.
Other formats are/will be available as modules.

Note: When implementing, try to work with a JSON-like
  tree and reuse `create_from_dict` and `HistogramBase.to_dict`.
"""
# from contextlib import suppress

from .json import load_json, parse_json, save_json
from .util import create_from_dict

__all__ = ["save_json", "load_json", "parse_json", "create_from_dict"]

# TODO: Re-enable ROOT
# Optional ROOT support
# with suppress(ImportError):
#     from .root import save_root, write_root  # noqa: F401
#
#     __all__.extend(["write_root", "save_root"])
