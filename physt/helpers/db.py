"""Helper functions to consume DB cursors."""
# TODO: Add tests with in-memory SQLite
# TODO: Implement non-adaptive
from typing import TYPE_CHECKING

from physt import h1, h2, h3

if TYPE_CHECKING:
    from typing import Tuple

    from physt.types import Histogram1D, Histogram2D, HistogramND


def _get_axis_names(cursor) -> Tuple[str, ...]:
    return tuple(field[0] for field in cursor.description)


def create_h1(cursor, *args, **kwargs) -> Histogram1D:
    axis_names = _get_axis_names(cursor)
    if len(axis_names) != 1:
        raise ValueError(f"Invalid number of columns: {len(axis_names)}")
    kwargs["axis_name"] = kwargs.get("axis_name", axis_names[0])
    if kwargs.get("adaptive", False):
        h = h1(None, *args, **kwargs)
        for row in cursor:
            h << row[0]
        return h
    else:
        raise NotImplementedError()


def create_h2(cursor, *args, **kwargs) -> Histogram2D:
    axis_names = _get_axis_names(cursor)
    if len(axis_names) != 2:
        raise ValueError(f"Invalid number of columns: {len(axis_names)}")
    kwargs["axis_names"] = kwargs.get("axis_names", axis_names)
    if kwargs.get("adaptive", False):
        h = h2(None, None, *args, **kwargs)
        for row in cursor:
            h << row
        return h
    else:
        raise NotImplementedError()


def create_h3(cursor, *args, **kwargs) -> HistogramND:
    # TODO: Refactor
    axis_names = _get_axis_names(cursor)
    if len(axis_names) != 3:
        raise ValueError(f"Invalid number of columns: {len(axis_names)}")
    kwargs["axis_names"] = kwargs.get("axis_names", axis_names)
    if kwargs.get("adaptive", False):
        h = h3(None, *args, **kwargs)
        for row in cursor:
            h << row
        return h
    else:
        raise NotImplementedError()
