"""Facade constructors to make creation of histograms easier."""
from collections import Iterable
from typing import Optional, Union

from .schema import build_schema, build_multi_schema
from .histogram import Histogram
from .array_utils import make_2d_array


def h1(a=None, *, schema=None, bins: Optional[Union[str, int,Iterable]]=None, weights=None, **kwargs) -> Histogram:
    # TODO: In passing arguments, take inspiration from scikit...
    schema = build_schema(schema, bins=bins, **kwargs)
    if a is not None:
        values = schema.fit_and_apply(a, weights=weights)
    else:
        values = None
    return Histogram(schema=schema, values=values)


def h2(a, *b, kind=None, weights=None, **kwargs):
    schema = build_multi_schema(kind, ndim=2, **kwargs)
    data = make_2d_array(2, a, *b)
    values = schema.fit_and_apply(data, weights=weights)
    return Histogram(schema=schema, values=values)


def h3(a, *bc, kind=None, **kwargs):
    raise NotImplementedError()


def h4(a, *bcd, kind=None, **kwargs):
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


# Namespace clean up
del (Iterable, Union, Optional)