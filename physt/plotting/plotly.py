"""Plot.ly backend for plotting in physt.

Currently, it uses matplotlib translation for 1D histograms:
- bar
- scatter
- line

TODO: More elaborate output planned
"""
from functools import wraps
from typing import Optional, Union

import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.graph_objs import Figure

from physt.histogram1d import Histogram1D
from physt.histogram_collection import HistogramCollection
from physt.histogram_nd import Histogram2D
from physt.util import pop_many
from .common import get_data, check_ndim

AbstractHistogram1D = Union[HistogramCollection, Histogram1D]
# TODO: Move this to the typing itself

DEFAULT_BARMODE = "overlay"
DEFAULT_ALPHA = 1.0


def enable_output(f):
    @wraps(f)
    def new_f(*args, write_to: Optional[str] = None, **kwargs) -> Figure:
        figure: Figure = f(*args, **kwargs)
        if write_to:
            pyo.plot(figure, filename=write_to)
        return figure
    return new_f


def enable_collection(f):
    """Decorator calling the wrapped function with a HistogramCollection as argument."""

    @wraps(f)
    def new_f(histogram: AbstractHistogram1D, **kwargs):
        if isinstance(histogram, HistogramCollection):
            return f(histogram, **kwargs)
        return f(HistogramCollection(histogram), **kwargs)

    return new_f


def _add_ticks(xaxis: go.layout.XAxis, histogram: AbstractHistogram1D, kwargs: dict):
    """Customize ticks for an axis (1D histogram)."""
    ticks = kwargs.pop("ticks", None)
    tick_handler = kwargs.pop("tick_handler", None)

    if tick_handler:
        if ticks:
            raise ValueError("Cannot specify both tick and tick_handler")
        ticks, labels = tick_handler(histogram, histogram.min_edge, histogram.max_edge)

        xaxis.tickvals = ticks
        xaxis.ticktext = labels

    elif ticks == "center":
        xaxis.tickvals = histogram.bin_centers
    elif ticks == "edge":
        xaxis.tickvals = histogram.bin_left_edges
    else:
        xaxis.tickvals = ticks


@enable_collection
def _line_or_scatter(h: HistogramCollection, *, mode: str, **kwargs):
    get_data_kwargs = pop_many(kwargs, "density", "cumulative", "flatten")
    data = [
        go.Scatter(
            x=histogram.bin_centers,
            y=get_data(histogram, **get_data_kwargs),
            mode=mode,
            name=histogram.name,
        )
        for histogram in h
    ]

    layout = go.Layout()

    _add_ticks(layout.xaxis, h[0], kwargs)

    figure = go.Figure(data=data, layout=layout)
    return figure


@enable_output
@check_ndim(1)
@enable_collection
def scatter(h: AbstractHistogram1D, **kwargs):
    return _line_or_scatter(h, mode="markers", **kwargs)


@enable_output
@check_ndim(1)
@enable_collection
def line(h: AbstractHistogram1D, **kwargs):
    return _line_or_scatter(h, mode="lines", **kwargs)


@enable_output
@check_ndim(1)
@enable_collection
def bar(
    h: HistogramCollection,
    *,
    barmode: str = DEFAULT_BARMODE,
    alpha: float = DEFAULT_ALPHA,
    **kwargs
):  # pylint: disable=blacklisted-name
    """Bar plot.

    Parameters
    ----------
    alpha: Opacity (0.0 - 1.0)
    barmode : "overlay" | "group" | "stack"
    """
    get_data_kwargs = pop_many(kwargs, "density", "cumulative", "flatten")
    data = [
        go.Bar(
            x=histogram.bin_centers,
            y=get_data(histogram, **get_data_kwargs),
            width=histogram.bin_widths,
            name=histogram.name,
            opacity=alpha,
            **kwargs
        )
        for histogram in h
    ]

    layout = go.Layout(barmode=barmode)

    _add_ticks(layout.xaxis, h[0], kwargs)

    figure = go.Figure(data=data, layout=layout)
    return figure


@enable_output
@check_ndim(2)
def map(h2: Histogram2D, **kwargs):
    """Heatmap."""
    data = [go.Heatmap(z=h2.frequencies, **kwargs)]
    layout = go.Layout()
    figure = go.Figure(data=data, layout=layout)
    return figure


types = ["bar", "scatter", "line"]
dims = {x: [1] for x in types}

types.append("map")
dims["map"] = [2]

# for plot_type in types:
#     if plot_type not in globals():
#         globals()[plot_type] = _wrap_matplotlib_f(mpl_backend.__dict__[plot_type])
