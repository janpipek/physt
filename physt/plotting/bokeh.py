"""
Bokeh backend for plotting in physt.

This is very basic.
"""
from __future__ import absolute_import
from bokeh.charts import show as bokeh_show
from bokeh.models import HoverTool, LinearColorMapper
from bokeh.models.sources import ColumnDataSource
import numpy as np

from .common import get_data

types = ("bar", "scatter", "map", "line")

dims = {
    "scatter" : [1],
    "line": [1],
    "bar" : [1],
    "map" : [2]
}


def _create_figure(histogram, **kwargs):
    title = histogram.title or "Histogram"
    axis_names = histogram.axis_names or ["x", "y"]
    if len(axis_names) == 1:
        axis_names = list(axis_names) + ["frequency"]

    from bokeh.plotting import figure
    return figure(tools="hover,save,pan,box_zoom,reset,wheel_zoom",
                  toolbar_location='above',
                  x_axis_label=axis_names[0], y_axis_label=axis_names[1],
                  title=title)


# All palette names that can be used in cmap argument
named_palettes = ["grey", "gray", "viridis", "magma", "inferno"]


def _line_scatter(h1, kind, show=True, **kwargs):
    """Line or scatter plot.
    
    The common functionality (the plots differ only in method called).
    """
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)
    size = kwargs.pop("size", 8)
    
    p = kwargs.pop("figure", _create_figure(h1))

    data = get_data(h1, cumulative=cumulative, density=density)

    plot_data = {
        "x" : h1.bin_centers,
        "y" : data
    }
    if kind == "line":
        p.line(plot_data['x'], plot_data['y'])
    elif kind == "scatter":
        p.scatter(plot_data['x'], plot_data['y'], size=size)
    if show:
        bokeh_show(p)
    return p


def line(h1, show=True, **kwargs):
    """Line plot."""
    return _line_scatter(h1=h1, kind="line", show=show, **kwargs)
    
    
def scatter(h1, show=True, **kwargs):
    """Scatter plot."""
    return _line_scatter(h1=h1, kind="scatter", show=show, **kwargs)


def bar(h1, show=True, **kwargs):
    """Bar chart."""
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)

    data = get_data(h1, cumulative=cumulative, density=density)

    plot_data = ColumnDataSource(data={
        "top" : data,
        "bottom" : np.zeros_like(data),
        "left" : h1.bin_left_edges,
        "right" : h1.bin_right_edges
    })
    p = kwargs.pop("figure", _create_figure(h1))
    p.quad(
        "left", "right", "top", "bottom",
        source=plot_data,
        color=kwargs.get("color", "blue"),
        line_width=kwargs.get("lw", 1),
        line_color=kwargs.get("line_color", "black"),
        fill_alpha=kwargs.get("alpha", 1),
        line_alpha=kwargs.get("alpha", 1),
    )
    p.select_one(HoverTool).tooltips = [
        ("bin", "@left..@right"),
        ("frequency", "@top")
    ]

    if show:
        bokeh_show(p)
    return p
    

def map(h2, show=True, cmap="gray", cmap_reverse=True, **kwargs):
    """Heat map."""
    density = kwargs.pop("density", False)
    data = get_data(h2, density=density)
    show_colorbar = kwargs.pop("show_colorbar", True)
    
    X, Y = np.meshgrid(h2.get_bin_centers(0), h2.get_bin_centers(1))
    dX, dY = np.meshgrid(h2.get_bin_widths(0), h2.get_bin_widths(1))
    
    source = ColumnDataSource({
        "frequency": data.T.flatten(),
        "x": X.flatten(),
        "y": Y.flatten(),
        "width": dX.flatten(),
        "height": dY.flatten()
    })
    
    import bokeh.palettes
    if cmap in named_palettes:
        palette_generator = getattr(bokeh.palettes, cmap)
        palette = palette_generator(256)
    elif cmap in bokeh.palettes.all_palettes:
        palette = bokeh.palettes.all_palettes[cmap]
    else:
        raise RuntimeError("Unknown palette")
    if cmap_reverse:
        palette = palette[::-1]
    
    mapper = LinearColorMapper(palette=palette, low=data.min(), high=data.max())
    
    p = kwargs.pop("figure", _create_figure(h2))
    
    p.rect(source=source,
           x="x", y="y", 
           width="width", height="height",
           fill_color={"field" : "frequency", "transform" : mapper},
           line_color="white")
    
    p.select_one(HoverTool).tooltips = [
        ("frequency", "@frequency")
    ]
    
    if show_colorbar:
        from bokeh.models import ColorBar, BasicTicker
        color_bar = ColorBar(color_mapper=mapper, #, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=6),
                         # formatter=PrintfTickFormatter(format="%d%%"),
                         # label_standoff=6, border_line_color=None,
                         location=(0, 0)
                         )
        p.add_layout(color_bar, 'right')
    
    if show:
        bokeh_show(p)
    return p
