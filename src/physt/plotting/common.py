"""
Functions that are shared by several (all) plotting backends.
"""
from __future__ import annotations

import re
from datetime import timedelta
from functools import wraps
from typing import TYPE_CHECKING

import numpy as np

from physt._bin_utils import (
    find_pretty_width_24,
    find_pretty_width_60,
    find_pretty_width_decimal,
)

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

    from physt.types import Histogram1D, HistogramBase

    TickCollection = Tuple[List[float], List[str]]
    """A tuple of tick values and labels."""


def get_data(
    histogram: HistogramBase,
    density: bool = False,
    cumulative: bool = False,
    flatten: bool = False,
) -> np.ndarray:
    """Get histogram data based on plotting parameters.

    Parameters
    ----------
    density : Whether to divide bin contents by bin size
    cumulative : Whether to return cumulative sums instead of individual
    flatten : Whether to flatten multidimensional bins
    """
    if density:
        if cumulative:
            data = (histogram / histogram.total).cumulative_frequencies
        else:
            data = histogram.densities
    else:
        if cumulative:
            try:
                data = histogram.cumulative_frequencies  # type: ignore
            except AttributeError:
                raise TypeError(
                    f"Type {type(histogram)} does not support cumulative frequencies."
                )
        else:
            data = histogram.frequencies

    if flatten:
        data = data.flatten()
    return data


def get_err_data(
    histogram: HistogramBase,
    density: bool = False,
    cumulative: bool = False,
    flatten: bool = False,
) -> np.ndarray:
    """Get histogram error data based on plotting parameters.

    Parameters
    ----------
    density : Whether to divide bin contents by bin size
    cumulative : Whether to return cumulative sums instead of individual
    flatten : Whether to flatten multidimensional bins
    """
    if cumulative:
        raise ValueError("Error bars not supported for cumulative plots.")
    if density:
        data = histogram.errors / histogram.bin_sizes
    else:
        data = histogram.errors
    if flatten:
        data = data.flatten()
    return data


def get_value_format(
    value_format: Union[Callable[[float], str], str, None],
) -> Callable[[float], str]:
    """Create a formatting function from a generic value_format argument.

    Parameters
    ----------
    value_format : A formatting function or a string.
    """
    if not value_format:
        return str

    if isinstance(value_format, str):
        format_str = "{0:" + value_format + "}"

        def value_format_(x):
            return format_str.format(x)

        return value_format_

    if callable(value_format):
        return value_format

    raise TypeError("`value_format` must be a string or a callable.")


def pop_kwargs_with_prefix(prefix: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pop all items from a dictionary that have keys beginning with a prefix.

    Parameters
    ----------
    prefix : The prefix to match.
    kwargs : The dictionary to be modified.

    Returns
    -------
    kwargs : Items popped from the original directory, with prefix removed.

    Note that the function modifies the original dictionary.
    """
    keys = [key for key in kwargs if key.startswith(prefix)]
    return {key[len(prefix) :]: kwargs.pop(key) for key in keys}


def check_ndim(ndim: Union[int, Tuple[int, ...]]) -> Callable[[Callable], Callable]:
    """Decorator checking proper histogram dimension.

    Parameters
    ----------
    ndim : The required dimensionality/ies of the histogram.
    """

    def wrapper(f):
        @wraps(f)
        def wrapped(h, *args, **kwargs):
            expected_dim = (ndim,) if isinstance(ndim, int) else ndim
            if h.ndim not in expected_dim:
                raise TypeError(
                    f"This type of plot must have dimension in {expected_dim}, {h.ndim} found."
                )
            return f(h, *args, **kwargs)

        return wrapped

    return wrapper


class TimeTickHandler:
    """Callable that creates ticks and labels corresponding to "sane" time values.

    Note: This class is very experimental and subject to change or disappear.
    """

    def __init__(self, level: Optional[str] = None):  # , format=None):
        self.level = self.parse_level(level) if level else None

    LEVELS = {
        "sec": 1,
        "min": 60,
        "hour": 3600,
        "day": 86400,
    }

    if TYPE_CHECKING:
        LevelType = Tuple[str, Union[float, int]]

    @classmethod
    def parse_level(
        cls, value: Union[LevelType, float, str, timedelta]
    ) -> "TimeTickHandler.LevelType":
        """Parse the level from any of the supported types."""
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f"Invalid level: {value}")
            if value[0] not in cls.LEVELS:
                raise ValueError(f"Invalid level: {value}")
            if not isinstance(value[1], (float, int)):
                raise ValueError(f"Invalid level: {value}")
            return value
        if isinstance(value, (float, int)):
            return cls.parse_level(timedelta(seconds=value))
        if isinstance(value, timedelta):
            # TODO: Implement
            raise NotImplementedError
        if isinstance(value, str):
            matchers = (
                ("^(center|edge)s?$", lambda m: (m[1], 0)),
                ("^([0-9\\.]+)?d(ay(s)?)?$", lambda m: ("day", float(m[1] or 1))),
                ("^([0-9]+)?h(our(s)?)?$", lambda m: ("hour", int(m[1] or 1))),
                ("^([0-9]+)?m(in(s)?)?$", lambda m: ("min", int(m[1] or 1))),
                (
                    "^([0-9\\.]+)?(\\.[0-9]+)?s(ec(s)?)?$",
                    lambda m: ("sec", float(m[1] or 1) + float("0." + (m[2] or "0"))),
                ),
            )
            for matcher in matchers:
                match = re.match(matcher[0], value)
                if match:
                    return matcher[1](match)
            raise ValueError(f"Cannot parse level: {value}")
        raise TypeError(f"Invalid level: {value}")

    @classmethod
    def deduce_level(cls, min_: float, max_: float) -> "TimeTickHandler.LevelType":
        """Determine the level of the tick handler from the histogram range."""
        ideal_width = (max_ - min_) / 6
        if ideal_width < 0.8:
            return ("sec", find_pretty_width_decimal(ideal_width))
        elif ideal_width < 50:
            return ("sec", find_pretty_width_60(ideal_width))
        elif ideal_width < 3000:
            return ("min", find_pretty_width_60(ideal_width / 60))
        elif ideal_width < 70000:
            return ("hour", find_pretty_width_24(ideal_width / 3600))
        else:
            return (
                "day",
                find_pretty_width_decimal(ideal_width / 86400),
            )  # # noqa: FURB126

    def get_time_ticks(
        self, h1: Histogram1D, level: LevelType, min_: float, max_: float
    ) -> List[float]:
        """Get ticks for a given level."""
        # TODO: Change to class method?
        if level[0] == "edge":
            return h1.numpy_bins.tolist()
        if level[0] == "center":
            return list(h1.bin_centers)

        width = level[1] * self.LEVELS[level[0]]
        min_factor = int(min_ // width)
        if min_ % width != 0:
            min_factor += 1
        max_factor = int(max_ // width)
        return list(np.arange(min_factor, max_factor + 1) * width)

    @classmethod
    def split_hms(cls, value: float) -> Tuple[bool, int, int, Union[int, float]]:
        """Split the time value into sign, hours, minutes, seconds"""
        value, negative = (value, False) if value >= 0 else (-value, True)
        hm, s = divmod(value, 60)
        h, m = (int(x) for x in divmod(hm, 60))
        s = s if s % 1 else int(s)
        return negative, h, m, s

    def format_time_ticks(self, ticks: List[float], level: LevelType) -> List[str]:
        if level[0] == "day":
            tick_days = [tick / 86400 for tick in ticks]
            if not any(tick % 1 for tick in tick_days):
                tick_days = [int(tick) for tick in tick_days]
            return [
                "{0} day{1}".format(tick, "" if tick == 1 else "s")
                for tick in tick_days
            ]
        else:
            hms = [self.split_hms(tick) for tick in ticks]
            include_hours = any(h for _, h, _, _ in hms)
            include_mins = any(h or m for _, h, m, _ in hms)
            include_secs = any(s != 0 for _, _, _, s in hms) or not include_hours
            sign = any(neg for neg, _, _, _ in hms)

            format_ = ""
            format_ += "{0}:" if include_hours else ""
            format_ += "{1}" if include_mins else ""
            format_ += ":" if include_mins and include_secs else ""
            format_ += "{2}" if include_secs else ""

            return [
                (("-" if neg else "+") if sign else "")
                + format_.format(
                    h,
                    m if not include_hours else str(m).zfill(2),
                    s if not include_mins else str(s).zfill(2),
                )
                for neg, h, m, s in hms
            ]

    def __call__(self, h1: Histogram1D, min_: float, max_: float) -> TickCollection:
        level = self.level or self.deduce_level(min_, max_)
        ticks = self.get_time_ticks(h1, level, min_, max_)
        tick_labels = self.format_time_ticks(ticks, level=level)
        return ticks, tick_labels
