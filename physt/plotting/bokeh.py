"""
Bokeh backend for plotting in physt.

This is very basic.
"""
from __future__ import absolute_import
from bokeh.plotting import figure
from bokeh.charts import Scatter
from bokeh.charts import show as bokeh_show
from bokeh.models import HoverTool
from bokeh.models.sources import ColumnDataSource
import numpy as np

from .common import get_data

types = ("bar", "scatter")

dims = {
    "scatter" : [1],
    "bar" : [1],
}


def scatter(h1, show=True, **kwargs):
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)

    data = get_data(h1, cumulative=cumulative, density=density)

    plot_data = {
        "x" : h1.bin_centers,
        "y" : data
    }
    p = Scatter(plot_data, x='x', y='y', **kwargs)
    if show:
        bokeh_show(p)
    return p


def bar(h1, show=True, **kwargs):
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)

    data = get_data(h1, cumulative=cumulative, density=density)

    plot_data = ColumnDataSource(data={
        "top" : data,
        "bottom" : np.zeros_like(data),
        "left" : h1.bin_left_edges,
        "right" : h1.bin_right_edges
    })
    tooltips = [
        ("bin", "@left..@right"),
        ("frequency", "@top")
    ]
    hover = HoverTool(tooltips=tooltips)
    p = kwargs.get("figure", figure(tools=[hover]))
    p.quad(
        "left", "right", "top", "bottom",
        source=plot_data,
        color=kwargs.get("color", "blue"),
        line_width=kwargs.get("lw", 1),
        line_color=kwargs.get("line_color", "black"),
        fill_alpha=kwargs.get("alpha", 1),
        line_alpha=kwargs.get("alpha", 1),
    )

    if show:
        bokeh_show(p)
    return p
