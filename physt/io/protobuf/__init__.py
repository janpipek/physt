"""Google's protocol buffer I/O support."""

from physt import __version__
from physt.histogram_base import HistogramBase
from physt.io import require_compatible_version
from .histogram_pb2 import Histogram as HistogramMessage
from .histogram_pb2 import Meta as MetaMessage

# Name of fields that are re-used from to_dict / from_dict
SIMPLE_CONVERSION_FIELDS = (
    "histogram_type", "dtype"
)

# Version writing the message
CURRENT_VERSION = __version__

# First version that should understand this
COMPATIBLE_VERSION = "0.3.42"


def save_protobuf(histogram):
    """Convert a histogram (or collection) to a message.

    Note: Currently, all binnings are converted to
      static form. When you load the histogram again,
      you will lose any related behaviour.
    
    Parameters
    ----------
    histogram : HistogramBase | list | dict
        Any histogram

    Returns
    -------
    message : google.protobuf.message.Message
        A protocol buffer message
    """
    
    histogram_dict = histogram.to_dict()
    message = HistogramMessage()

    for field in SIMPLE_CONVERSION_FIELDS:
        setattr(message, field, histogram_dict[field])

    # All meta data
    PREDEFINED = ("name", "title", "axis_names")
    meta_message = MetaMessage()
    user_defined = {}
    for key, value in histogram.meta_data.items():
        if key not in PREDEFINED:
            user_defined[str(key)] = str(value)
    for key in PREDEFINED:
        if key in histogram.meta_data:
            setattr(meta_message, key, str(histogram.meta_data[key]))

    message.physt_version = CURRENT_VERSION
    message.physt_compatible = COMPATIBLE_VERSION
    return message

def _load_v0342(message):
    pass

def load_protobuf(message):


    # Currently the only implementation
    return _load_v0342(message)
