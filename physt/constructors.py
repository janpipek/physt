from .schema import build_schema, build_multi_schema
from .array_utils import make_2d_array
from .histogram import Histogram


DEFAULT_HISTOGRAM_TYPE = "human"


def h1(a, kind=DEFAULT_HISTOGRAM_TYPE, *, weights=None, **kwargs):
    schema = build_schema(kind, **kwargs)
    values = schema.fit_and_apply(a, weights=weights)
    return Histogram(schema=schema, values=values)


def h2(a, *b, kind=DEFAULT_HISTOGRAM_TYPE, weights=None, **kwargs):
    schema = build_multi_schema(kind, ndim=2, **kwargs)
    data = make_2d_array(2, a, *b)
    values = schema.fit_and_apply(data, weights=weights)
    return Histogram(schema=schema, values=values)


def h3(a, *bc, kind=DEFAULT_HISTOGRAM_TYPE, **kwargs):
    raise NotImplementedError()


def h4(a, *bcd, kind=DEFAULT_HISTOGRAM_TYPE, **kwargs):
    raise NotImplementedError()


def h5():
    raise NotImplementedError()


def h6():
    raise NotImplementedError()


def h7():
    raise NotImplementedError()


def h8():
    raise NotImplementedError()


def h9():
    raise NotImplementedError()