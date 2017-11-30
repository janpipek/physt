
from physt.histogram import Histogram
from physt.builder import HistogramBuilder
from physt.schema import NumpySchema, StaticSchema

__all__ = ["histogram"]


def histogram(a, bins=10, range=None, normed=False, weights=None, density=None) -> Histogram:
    if isinstance(bins, (str, int)):
        schema = NumpySchema(bins, range=range)
    else:
        schema = StaticSchema(edges=bins)
    builder = HistogramBuilder((schema,))
    h = builder(a, weights=weights)
    if normed:
        h.normalize(inplace=True)    # TODO: Implement
    if density:
        # TODO: Probably warn
        pass
    return h

def histogram2d():
    pass

def histogramdd():
    pass
