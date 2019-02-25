"""
Functions that are shared by several (all) plotting backends.

"""
import re
from typing import Tuple, List, Union, Callable
from datetime import timedelta, time

import numpy as np

from physt.histogram_base import HistogramBase
from physt.histogram1d import Histogram1D


def get_data(histogram: HistogramBase, density: bool = False, cumulative: bool = False, flatten: bool = False) -> np.ndarray:
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
            data = histogram.cumulative_frequencies
        else:
            data = histogram.frequencies

    if flatten:
        data = data.flatten()
    return data


def get_err_data(histogram: HistogramBase, density: bool = False, cumulative: bool = False, flatten: bool = False) -> np.ndarray:
    """Get histogram error data based on plotting parameters.

    Parameters
    ----------
    density : Whether to divide bin contents by bin size
    cumulative : Whether to return cumulative sums instead of individual
    flatten : Whether to flatten multidimensional bins
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


def get_value_format(value_format: Union[Callable, str] = str) -> Callable[[float], str]:
    """Create a formatting function from a generic value_format argument.
    """
    if value_format is None:
        value_format = ""
    if isinstance(value_format, str):
        format_str = "{0:" + value_format + "}"

        def value_format(x): return format_str.format(x)

    return value_format


def pop_kwargs_with_prefix(prefix: str, kwargs: dict) -> dict:
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
    """Callable that creates ticks and labels corresponding to "sane" time values.

    Note: This class is very experimental and subject to change or disappear.
    """

    def __init__(self, level: str = None): #, format=None):
        self.level = self.parse_level(level) if level else None
        # self.format = format  # TODO: Really?

    LEVELS = {
        "sec": 1,
        "min": 60,
        "hour": 3600,
    }

    LevelType = Tuple[str, Union[float, int]]

    @classmethod
    def parse_level(cls, value: Union[LevelType, float, str, timedelta]) -> LevelType:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Invalid level: {0}".format(value))
            if value[0] not in cls.LEVELS:
                raise ValueError("Invalid level: {0}".format(value))
            if not isinstance(value[1], (float, int)):
                raise ValueError("Invalid level: {0}".format(value))
            return value
        elif isinstance(value, (float, int)):
            return cls.parse_level(timedelta(seconds=value))
        elif isinstance(value, timedelta):
            ... # TODO: Implement
        elif isinstance(value, str):
            matchers = (
                ("^(center|edge)s?$", lambda m: (m[1], 0)),
                ("^([0-9]+)?h(our(s)?)?$", lambda m: ("hour", int(m[1] or 1))),
                ("^([0-9]+)?m(in(s)?)?$", lambda m: ("min", int(m[1] or 1))),
                ("^([0-9\.]+)?(\.[0-9]+)?s(ec(s)?)?$", lambda m: ("sec",
                                                                  float(m[1] or 1) + float("0." + (m[2] or "0")))),
            )
            for matcher in matchers:
                match = re.match(matcher[0], value)
                if match:
                    return matcher[1](match)
            raise ValueError("Cannot parse level: {0}".format(value))
        else:
            raise ValueError("Invalid level: {0}".format(value))

    @classmethod
    def find_human_width_decimal(cls, raw_width: float) -> float:
        subscales = np.array([0.5, 1, 2, 2.5, 5, 10])
        power = np.floor(np.log10(raw_width)).astype(int)
        best_index = np.argmin(np.abs(np.log(subscales * (10.0 ** power) / raw_width)))
        return (10.0 ** power) * subscales[best_index]

    @classmethod
    def find_human_width_60(cls, raw_width: float) -> int:
        subscales = (1, 2, 5, 10, 15, 20, 30,)
        best_index = np.argmin(np.abs(np.log(subscales / raw_width)))
        return subscales[best_index]     

    @classmethod
    def deduce_level(cls, h1: Histogram1D, min_: float, max_: float) -> LevelType:
        ideal_width = (max_ - min_) / 6
        if ideal_width < 0.8:
            return ("sec", cls.find_human_width_decimal(ideal_width))
        elif ideal_width < 50:
            return ("sec", cls.find_human_width_60(ideal_width))
        elif ideal_width < 3000:
            return ("min", cls.find_human_width_60(ideal_width / 60))
        else:
            return ("hour", cls.find_human_width_decimal(ideal_width / 3600))

    def get_time_ticks(self, h1: Histogram1D, level: LevelType, min_: float, max_: float) -> List[float]:
        # TODO: Change to class method?
        if level[0] == "edge":
            return h1.numpy_bins.tolist()
        elif level[0] == "center":
            return h1.bin_centers
        else:
            width = level[1] * self.LEVELS[level[0]]
            min_factor = int(min_ // width)
            if min_ % width != 0:
                min_factor += 1
            max_factor = int(max_ // width)
            return list(np.arange(min_factor, max_factor + 1) * width)

    @classmethod
    def split_hms(cls, value) -> Tuple[bool, int, int, Union[int, float]]:
        value, negative = (value, False) if value >= 0 else (-value, True)
        hm, s = divmod(value, 60)
        h, m = (int(x) for x in divmod(hm, 60))
        s = s if s % 1 else int(s)
        return negative, h, m, s

    def format_time_ticks(self, ticks: List[float]) -> List[str]:
        hms = [self.split_hms(tick) for tick in ticks]
        include_hours = any(h for _, h, _, _ in hms)
        include_mins = any(h or m for _, h, m, _ in hms)
        include_secs = any(s != 0 for _, _, _, s in hms) or not include_hours
        secs_float = any(s % 1 for _, _, _, s in hms)
        sign = any(neg for neg, _, _, _ in hms)

        format = ""
        format += "{0}:" if include_hours else ""
        format += "{1}" if include_mins else ""
        format += ":" if include_mins and include_secs else ""
        format += "{2}" if include_secs else ""

        return [
            (("-" if neg else "+") if sign else "") +
            format.format(
                h,
                m if not include_hours else str(m).zfill(2),
                s if not include_mins else str(s).zfill(2)
            )
            for neg, h, m, s in hms]

    def __call__(self, h1: Histogram1D, min_: float, max_: float) -> TickCollection:
        level = self.level or self.deduce_level(h1, min_, max_)
        ticks = self.get_time_ticks(h1, level, min_, max_)
        tick_labels = self.format_time_ticks(ticks)
        return ticks, tick_labels
