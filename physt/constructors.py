"""Facade constructors to make creation of histograms easier."""
from collections import Iterable
from typing import Optional, Union

from .schema import build_schema
from .histogram import Histogram



def h1(a=None, *, schema=None, bins: Optional[Union[str, int,Iterable]]=None, weights=None, **kwargs) -> Histogram:
    # TODO: In passing arguments, take inspiration from scikit...
    schema = build_schema(schema, bins=bins, **kwargs)
    if a is not None:
        values = schema.fit_and_apply(a, weights=weights)
    else:
        values = None
    return Histogram(schema=schema, values=values)


def h2(a=None, *b, kind="human", **kwargs) -> Histogram:
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


# Namespace clean up
del (Iterable, Union, Optional)