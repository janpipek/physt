"""
Bokeh backend for plotting in physt.

This is very basic.
"""
from __future__ import absolute_import
from bokeh.plotting import figure
from bokeh.charts import Scatter
from bokeh.charts import show as bokeh_show
from bokeh.models import HoverTool, LinearColorMapper
from bokeh.models.sources import ColumnDataSource
import numpy as np

from .common import get_data

types = ("bar", "scatter", "map")

dims = {
    "scatter" : [1],
    "bar" : [1],
    "map" : [2]
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
    
def map(h2, show=True, **kwargs):
    density = kwargs.pop("density", False)
    data = get_data(h2, density=density)
    
    source = ColumnDataSource({
        "color": data.T.flatten()
    })
    
    X, Y = np.meshgrid(h2.get_bin_left_edges(0), h2.get_bin_left_edges(1))
    dX, dY = np.meshgrid(h2.get_bin_widths(0), h2.get_bin_widths(1))
    #colors = ["#75968f", "#550b1d"]
    from bokeh.palettes import Greys256
    mapper = LinearColorMapper(palette=Greys256, low=data.max(), high=data.min())
    
    p = kwargs.get("figure", figure())
    # TODO: Use ColumnDataSource fields
    p.rect(x=Y.flatten(), y=X.flatten(), source=source,
           width=dX.flatten(), height=dY.flatten(),
           fill_color={"field" : "color", "transform" : mapper},
           line_color="white")
    
    if show:
        bokeh_show(p)
    return p
