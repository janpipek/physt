"""
Funct;ions that are shared by several (all) plotting backends.

"""
import re
from typing import Tuple, List, Union
from datetime import timedelta, time

import numpy as np

from physt.histogram1d import Histogram1D


def get_data(histogram, density=False, cumulative=False, flatten=False):
    """Get histogram data based on plotting parameters.

    Parameters
    ----------
    h : physt.histogram_base.HistogramBase
    density : bool
        Whether to divide bin contents by bin size
    cumulative : bool
        Whether to return cumulative sums instead of individual
    flatten : bool
        Whether to flatten multidimensional bins

    Returns
    -------
    np.ndarray

    """
    if density:
        if cumulative:
            data = (histogram / histogram.total).cumulative_frequencies
        else:
            data = histogram.densities
    else:
        if cumulative:
            data = histogram.cumulative_frequencies
        else:
            data = histogram.frequencies

    if flatten:
        data = data.flatten()
    return data


def get_err_data(histogram, density=False, cumulative=False, flatten=False):
    """Get histogram error data based on plotting parameters.

    Parameters
    ----------
    h : physt.histogram_base.HistogramBase
    density : bool
        Whether to divide bin contents by bin size
    cumulative : bool
        Whether to return cumulative sums instead of individual
    flatten : bool
        Whether to flatten multidimensional bins

    Returns
    -------
    np.ndarray
    """
    if cumulative:
        raise RuntimeError("Error bars not supported for cumulative plots.")
    if density:
        data = histogram.errors / histogram.bin_sizes
    else:
        data = histogram.errors
    if flatten:
        data = data.flatten()
    return data


def get_value_format(value_format=str):
    """Create a formatting function from a generic value_format argument.
    
    Parameters
    ----------
    value_format : str or Callable

    Returns
    -------
    Callable
    """
    if value_format is None:
        value_format = ""
    if isinstance(value_format, str):
        format_str = "{0:" + value_format + "}"
        value_format = lambda x: format_str.format(x)
    
    return value_format


def pop_kwargs_with_prefix(prefix, kwargs):
    """Pop all items from a dictionary that have keys beginning with a prefix.

    Parameters
    ----------
    prefix : str
    kwargs : dict

    Returns
    -------
    kwargs : dict
        Items popped from the original directory, with prefix removed.
    """
    keys = [key for key in kwargs if key.startswith(prefix)]
    return {key[len(prefix):]: kwargs.pop(key) for key in keys}


TickCollection = Tuple[List[float], List[str]]


class TimeTickHandler:
    """

    Note: This class is very experimental and subject to change or disappear.
    """

    def __init__(self, level=None, format=None):
        self.level = self.parse_level(level) if level else None
        self.format = format

    LEVELS = {
        "sec": 1,
        "min": 60,
        "hour": 3600,
    }

    LevelType = Tuple[str, int]

    @classmethod
    def parse_level(cls, value: Union[LevelType, float, str, timedelta]) -> LevelType:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Invalid level: {0}".format(value))
            if value[0] not in cls.LEVELS:
                raise ValueError("Invalid level: {0}".format(value))
            if not isinstance(value[1], (float, int)):
                raise ValueError("Invalid level: {0}".format(vaiue))
            return value
        elif isinstance(value, (float, int)):
            ...
        elif isinstance(value, timedelta):
            ...
        elif isinstance(value, str):
            matchers = (
                ("^([0-9]+)?h(our(s)?)?$", lambda m : ("hour", int(m[1] or 1))),
                ("^([0-9]+)?m(in(s)?)?$", lambda m : ("min", int(m[1] or 1))),
                ("^([0-9]+)?(\.[0-9]+)?s(ec(s)?)?$", lambda m : ("sec", float(m[1] or 1) + float("0." + (m[2] or "0")))),
            )
            for matcher in matchers:
                match = re.match(matcher[0], value)
                if match:
                    return matcher[1](match)
            raise ValueError("Cannot parse level: {0}".format(value))
        else:
            raise ValueError("Invalid level: {0}".format(vaiue))

    @classmethod
    def deduce_level(cls, h1: Histogram1D) -> str:
        return ("min", 1)
        # TODO: really?

    def get_time_ticks(self, h1: Histogram1D, level, min_: float, max_: float) -> List[float]:
        width = level[1] * self.LEVELS[level[0]]
        min_factor = int(min_ // width)
        if min_ % width != 0:
            min_factor += 1
        max_factor = int(max_ // width)
        return list(np.arange(min_factor, max_factor + 1) * width)
        
    @classmethod
    def split_hms(cls, value) -> Tuple[int, int, float]:
        ...
    
    def format_time_ticks(self, ticks: List[float]) -> List[str]:
        ...
        # return [str(tick) for tick in ticks]
        
        # deltas = [self.split_hms(tick) for tick in ticks]
        # if self.format:
        #     format = self.format
        # else:
        #     include_micros = any(delta.microseconds for delta in deltas)
        #     include_secs = any(delta.seconds % 60 for delta in deltas) or include_micros
        #     include_hours = any(delta.total_seconds() >= 3600 for delta in deltas)
        #     include_minutes = include_hours or any(delta.total_seconds() >= 60 for delta in deltas)
        #     # format = "%H:%M:%S"
        #     format = ""
        #     format += "%H:" if include_hours else ""
        #     format += "%M" if include_minutes else ""
        #     format += ":%S" if include_secs else ""
        #     format += ".%f" if include_micros else ""
        #     format += "}"
        # return [format.format(time(delta)) for delta in deltas]

    def __call__(self, h1: Histogram1D, min_: float, max_: float) -> TickCollection:
        level = self.level or cls.deduce_level(h1)
        ticks = self.get_time_ticks(h1, level, min_, max_)
        tick_labels = self.format_time_ticks(ticks)
        return ticks, tick_labels 
