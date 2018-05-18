from .schema import build_schema
from .histogram import Histogram


def h1(a, kind="human", *, weights=None, **kwargs):
    schema = build_schema(kind, **kwargs)
    values = schema.fit_and_apply(a, weights=weights)
    return Histogram(schema=schema, values=values)


def h2(a, *b, kind="human", **kwargs):
    raise NotImplementedError()


def h3(a, *bc, kind="human", **kwargs):
    raise NotImplementedError()


def h4(a, *bcd, kind="human", **kwargs):
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