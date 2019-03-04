"""Plot.ly backend for plotting in physt.

Currently, it uses matplotlib translation for 1D histograms:
- bar
- scatter
- line

TODO: More elaborate output planned
"""
from functools import wraps
from typing import Any, Optional, Union

import plotly.offline as pyo
import plotly.plotly as pyp
import plotly.graph_objs as go

from physt.histogram1d import Histogram1D, HistogramBase
from physt.histogram_nd import Histogram2D
from physt.histogram_collection import HistogramCollection
from physt.util import pop_many
from . import matplotlib as mpl_backend
from .common import get_data


AbstractHistogram1D = Union[HistogramCollection, Histogram1D]


DEFAULT_BARMODE = "overlay"
DEFAULT_ALPHA = 1.0


def _wrap_standard_f(f):
    @wraps(f)
    def new_f(*args, raw=False, offline=True, write_to=None, **kwargs):
        py = pyo if offline else pyp
        object = f(*args, **kwargs)
        if raw:
            return object
        if write_to:
            return py.plot(object, filename=write_to)
        else:
            return py.iplot(object)

    return new_f


def _wrap_matplotlib_f(f):
    @wraps(f)
    def new_f(*args, offline=True, write_to=None, **kwargs):
        py = pyo if offline else pyp
        ax = f(*args, **kwargs)
        fig = ax.figure
        if write_to:
            return py.plot_mpl(fig, filename=write_to)
        else:
            return py.iplot_mpl(fig)

    return new_f


def wrap(*, mpl_function: Optional[Any] = None):
    def decorate(function):
        normal_variant = _wrap_standard_f(function)
        if mpl_function:
            mpl_variant = _wrap_matplotlib_f(mpl_function)

            @wraps(function)
            def new_f(*args, mpl: bool = False, **kwargs):
                if mpl:
                    return mpl_variant(*args, **kwargs)
                else:
                    return normal_variant(*args, **kwargs)
            return new_f
        else:
            return normal_variant

    return decorate


def enable_collection(f):
    """Call the wrapped function with a HistogramCollection as argument."""
    @wraps(f)
    def new_f(h: AbstractHistogram1D, **kwargs):
        from physt.histogram_collection import HistogramCollection
        if isinstance(h, HistogramCollection):
            return f(h, **kwargs)
        else:
            return f(HistogramCollection(h), **kwargs)
    return new_f


def _add_ticks(xaxis: go.layout.XAxis, histogram: HistogramBase, kwargs: dict):
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
def _line_or_scatter(h: AbstractHistogram1D, *, mode: str, **kwargs):
    get_data_kwargs = pop_many(kwargs, "density", "cumulative", "flatten")
    data = [go.Scatter(
        x=histogram.bin_centers,
        y=get_data(histogram, **get_data_kwargs),
        mode=mode,
        name=histogram.name
    ) for histogram in h]

    layout = go.Layout()

    _add_ticks(layout.xaxis, h[0], kwargs)

    figure = go.Figure(data=data, layout=layout)
    return figure


@wrap(mpl_function=mpl_backend.scatter)
def scatter(h: AbstractHistogram1D, **kwargs):
    return _line_or_scatter(h, mode="markers", **kwargs)


@wrap(mpl_function=mpl_backend.line)
def line(h: AbstractHistogram1D, **kwargs):
    return _line_or_scatter(h, mode="lines", **kwargs)


@wrap(mpl_function=mpl_backend.bar)
@enable_collection
def bar(h: Histogram2D, *,
        barmode: str = DEFAULT_BARMODE,
        alpha: float = DEFAULT_ALPHA,
        **kwargs):
    """Bar plot.

    Parameters
    ----------
    alpha: Opacity (0.0 - 1.0)
    barmode : "overlay" | "group" | "stack"
    """
    get_data_kwargs = pop_many(kwargs, "density", "cumulative", "flatten")
    data = [go.Bar(
        x=histogram.bin_centers,
        y=get_data(histogram, **get_data_kwargs),
        width=histogram.bin_widths,
        name=histogram.name,
        opacity=alpha,
        **kwargs
    ) for histogram in h]

    layout = go.Layout(barmode=barmode)

    _add_ticks(layout.xaxis, h[0], kwargs)

    figure = go.Figure(data=data, layout=layout)
    return figure


@wrap()
def map(h2: Histogram2D,
        **kwargs):
    """Heatmap.

    """
    data = [go.Heatmap(z=h2.frequencies, **kwargs)]
    layout = go.Layout()
    figure = go.Figure(data=data, layout=layout)
    return figure


types = ["bar", "scatter", "line"]
dims = {x:[1] for x in types}

types.append("map")
dims["map"] = [2]

# for plot_type in types:
#     if plot_type not in globals():
#         globals()[plot_type] = _wrap_matplotlib_f(mpl_backend.__dict__[plot_type])
    

