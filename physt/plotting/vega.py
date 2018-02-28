"""Vega backend for plotting in physt.
"""
from __future__ import absolute_import

import codecs
import json
from functools import wraps

from .common import get_data

VEGA_IPYTHON_PLUGIN_ENABLED = False
VEGA_ERROR = None

# Options
DEFAULT_WIDTH = 400
DEFAULT_HEIGHT = 200
DEFAULT_PADDING = 5

PALETTES = [
    "Viridis",
    "Magma",
    "Inferno",
    "Plasma",
    "Blues",
    "Greens",
    "Greys",
    "Purples",
    "Reds",
    "Oranges",
    "BlueOrange",
    "BrownBlueGreen",
    "PurpleGreen",
    "PinkYellowGreen",
    "PurpleOrange",
    "RedBlue",
    "RedGrey",
    "RedYellowBlue",
    "RedYellowGreen",
    "BlueGreen",
    "BluePurple",
    "GreenBlue",
    "OrangeRed",
    "PurpleBlueGreen",
    "PurpleBlue",
    "PurpleRed",
    "RedPurple",
    "YellowGreenBlue",
    "YellowGreen",
    "YellowOrangeBrown",
    "YellowOrangeRed"
]
DEFAULT_PALETTE = PALETTES[0]


# Hack to find whether we can display inline images in IPython notebook
try:
    from IPython import get_ipython

    if get_ipython():
        try:
            import vega3

            VEGA_IPYTHON_PLUGIN_ENABLED = True
        except ImportError:
            VEGA_ERROR = "Library 'vega3' not present"
    else:
        VEGA_ERROR = "Not in a an interactive IPython shell."
except:
    VEGA_ERROR = "IPython not installed."

# Declare supported plot types.
types = ("bar", "scatter", "line", "map", "map_with_slider")
dims = {
    "bar": [1],
    "scatter": [1],
    "line": [1],
    "map": [2],
    "map_with_slider": [3],
}


def enable_inline_view(f):
    """Decorator to enable in-line viewing in Python and saving to external file.

    It adds several parameters to each decorated plotted function:

    Parameters
    ----------
    write_to: str (optional)
        Path to write vega JSON to.
    display: "auto" | True | False
        Whether to try in-line display in IPython
    indent: int
        Indentation of JSON
    """
    @wraps(f)
    def wrapper(hist, write_to=None, display="auto", indent=2, **kwargs):

        vega_data = f(hist, **kwargs)

        if display is True and not VEGA_IPYTHON_PLUGIN_ENABLED:
            raise RuntimeError("Cannot display vega plot: {0}".format(VEGA_ERROR))

        if display == "auto":
            display = write_to is None

        if write_to is not None:
            with codecs.open(write_to, "w", encoding="utf-8") as out:
                json.dump(vega_data, out, indent=indent)

        if VEGA_IPYTHON_PLUGIN_ENABLED and display:
            from vega3 import Vega
            return Vega(vega_data)
        else:
            return vega_data

    return wrapper


@enable_inline_view
def bar(h1, **kwargs):
    """Bar plot of 1D histogram.

    Parameters
    ----------
    h1 : physt.histogram1d.Histogram1D
        Dimensionality of histogram for which it is applicable
    """
    vega = _create_figure(kwargs)
    _add_title(h1, vega, kwargs)
    _create_scales(h1, vega, kwargs)
    _create_axes(h1, vega, kwargs)

    data = get_data(h1, kwargs.pop("density", None), kwargs.pop("cumulative", None)).tolist()
    lefts = h1.bin_left_edges.tolist()
    rights = h1.bin_right_edges.tolist()

    vega["data"] = [{
        "name": "table",
        "values": [{
                "x": lefts[i],
                "x2": rights[i],
                "y": data[i],
            }
            for i in range(h1.bin_count)
        ]
    }]

    vega["marks"] = [
        {
            "type": "rect",
            "from": {"data": "table"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "x"},
                    "x2": {"scale": "xscale", "field": "x2"},
                    "y": {"scale": "yscale", "value": 0},
                    "y2": {"scale": "yscale", "field": "y"},
                    # "stroke": {"scale": "color", "field": "c"},
                    "strokeWidth": {"value": 2}
                },
                "update": {
                    "fillOpacity": {"value": 1}
                },
                "hover": {
                     "fillOpacity": {"value": 0.5}
                }
            }
        }
    ]

    return vega


@enable_inline_view
def scatter(h1, **kwargs):
    """Scatter plot of 1D histogram values.

    Points are horizontally placed in bin centers.

    Parameters
    ----------
    h1 : physt.histogram1d.Histogram1D
        Dimensionality of histogram for which it is applicable
    """
    vega = _scatter_or_line(h1, kwargs)
    vega["marks"] = [
        {
            "type": "symbol",
            "from": {"data": "table"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    "shape": {"value": "circle"},
                    # "stroke": {"scale": "color", "field": "c"},
                    "strokeWidth": {"value": 2}
                },
                # "update": {
                #     "interpolate": {"signal": "interpolate"},
                #     "fillOpacity": {"value": 1}
                # },
                # "hover": {
                #     "fillOpacity": {"value": 0.5}
                # }
            }
        }
    ]
    return vega


@enable_inline_view
def line(h1, **kwargs):
    """Line plot of 1D histogram values.

    Points are horizontally placed in bin centers.

    Parameters
    ----------
    h1 : physt.histogram1d.Histogram1D
        Dimensionality of histogram for which it is applicable
    """
    vega = _scatter_or_line(h1, kwargs)
    vega["marks"] = [
        {
            "type": "line",
            "from": {"data": "table"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    # "stroke": {"scale": "color", "field": "c"},
                    "strokeWidth": {"value": 2}
                },
                # "update": {
                #     "interpolate": {"signal": "interpolate"},
                #     "fillOpacity": {"value": 1}
                # },
                # "hover": {
                #     "fillOpacity": {"value": 0.5}
                # }
            }
        }
    ]
    return vega


@enable_inline_view
def map(h2, show_zero=True, show_values=False, **kwargs):
    """Heat-map of two-dimensional histogram.

    Parameters
    ----------
    h2 : physt.histogram_nd.Histogram2D
        Dimensionality of histogram for which it is applicable
    show_zero : bool
    show_values : bool
    """
    vega = _create_figure(kwargs)
    cmap = kwargs.pop("cmap", DEFAULT_PALETTE)

    values = get_data(h2, kwargs.pop("density", None), kwargs.pop("cumulative", None)).tolist()

    _add_title(h2, vega, kwargs)
    _create_scales(h2, vega, kwargs)
    _create_axes(h2, vega, kwargs)

    vega["scales"].append(
        {
            "name": "color",
            "type": "sequential",
            "range": {"scheme": cmap},
            "domain": {"data": "table", "field": "c"},
            "zero": False, "nice": False
        }
    )

    x = h2.get_bin_centers(0)
    y = h2.get_bin_centers(1)
    x1 = h2.get_bin_left_edges(0)
    x2 = h2.get_bin_right_edges(0)
    y1 = h2.get_bin_left_edges(1)
    y2 = h2.get_bin_right_edges(1)

    data = []
    for i in range(h2.shape[0]):
        for j in range(h2.shape[1]):
            if not show_zero and values[i][j] == 0:
                continue
            data.append({
                "x": x[i],
                "x1": x1[i],
                "x2": x2[i],
                "y": y[j],
                "y1": y1[j],
                "y2": y2[j],
                "c": values[i][j],
            })

    vega["data"] = [{
        "name": "table",
        "values": data
    }]

    vega["marks"] = [
        {
            "type": "rect",
            "from": {"data": "table"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "x1"},
                    "x2": {"scale": "xscale", "field": "x2"},
                    "y": {"scale": "yscale", "field": "y1"},
                    "y2": {"scale": "yscale", "field": "y2"},
                    "fill": {"scale": "color", "field": "c"},
                    "stroke": {"value": 0},
                    # "strokeWidth": {"value": 0},
                    # "fillColor": {"value": "#ffff00"}
                },
                # "update": {
                #     "fillOpacity": {"value": 0.6}
                # },
                # "hover": {
                #     "fillOpacity": {"value": 0.5}
                # }
            }
        }
    ]

    if show_values:
        vega["marks"].append(
            {
                "type": "text",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "fontSize": {"value": 13},
                        "fontWeight": {"value": "bold"},
                        "text": {"field": "c"},
                        "x": {"scale": "xscale", "field": "x"},
                        "y": {"scale": "yscale", "field": "y"},
                    }
                }
            }
        )

    return vega


@enable_inline_view
def map_with_slider(h3, show_zero=True, show_values=False, **kwargs):
    """Heatmap showing slice in first two dimensions, third dimension represented as a slider.

    Parameters
    ----------
    h3 : physt.histogram_nd.HistogramND
        A three-dimensional diagram to plot.
    show_zero : bool
    show_values : bool
    """
    vega = _create_figure(kwargs)
    cmap = kwargs.pop("cmap", DEFAULT_PALETTE)

    values_arr = get_data(h3, kwargs.pop("density", None), kwargs.pop("cumulative", None))
    values = values_arr.tolist()

    _add_title(h3, vega, kwargs)
    _create_scales(h3, vega, kwargs)
    _create_axes(h3, vega, kwargs)

    vega["scales"].append(
        {
            "name": "color",
            "type": "sequential",
            "domain": [float(values_arr.min()), float(values_arr.max())],
            "range": {"scheme": cmap},
            "zero": False,
            "nice": False
        }
    )

    x = h3.get_bin_centers(0)
    y = h3.get_bin_centers(1)
    x1 = h3.get_bin_left_edges(0)
    x2 = h3.get_bin_right_edges(0)
    y1 = h3.get_bin_left_edges(1)
    y2 = h3.get_bin_right_edges(1)

    data = []
    for i in range(h3.shape[0]):
        for j in range(h3.shape[1]):
            for k in range(h3.shape[2]):
                if not show_zero and values[i][j][k] == 0:
                    continue
                data.append({
                    "x": x[i],
                    "x1": x1[i],
                    "x2": x2[i],
                    "y": y[j],
                    "y1": y1[j],
                    "y2": y2[j],
                    "k": k,
                    "c": values[i][j][k],
                })

    vega["signals"] = [
        { "name": h3.axis_names[2], "value": h3.shape[2] // 2,
          "bind": {"input": "range", "min": 0, "max": h3.shape[2] - 1, "step": 1} }
    ]

    vega["legends"] = [
        {"fill": "color", "type": "gradient"}
    ]

    vega["data"] = [{
        "name": "table",
        "values": data,
        "transform": [
             {
                 "type": "filter",
                 "expr": "z == datum.k",
             }
        ]
    }]

    vega["marks"] = [
        {
            "type": "rect",
            "from": {"data": "table"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "x1"},
                    "x2": {"scale": "xscale", "field": "x2"},
                    "y": {"scale": "yscale", "field": "y1"},
                    "y2": {"scale": "yscale", "field": "y2"},
                    "fill": {"scale": "color", "field": "c"},
                    "stroke": {"value": 0},
                    # "strokeWidth": {"value": 0},
                    # "fillColor": {"value": "#ffff00"}
                },
                # "update": {
                #     "fillOpacity": {"value": 0.6}
                # },
                # "hover": {
                #     "fillOpacity": {"value": 0.5}
                # }
            }
        }
    ]

    if show_values:
        vega["marks"].append(
            {
                "type": "text",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "fontSize": {"value": 13},
                        "fontWeight": {"value": "bold"},
                        "text": {"field": "c"},
                        "x": {"scale": "xscale", "field": "x"},
                        "y": {"scale": "yscale", "field": "y"},
                    }
                }
            }
        )

    return vega


def _scatter_or_line(h1, kwargs):
    """

    Parameters
    ----------
    h1 : physt.histogram1d.Histogram1D
        Dimensionality of histogram for which it is applicable
    kwargs : dict
    """
    vega = _create_figure(kwargs)
    data = get_data(h1, kwargs.pop("density", None), kwargs.pop("cumulative", None)).tolist()
    centers = h1.bin_centers.tolist()

    vega["data"] = [{
        "name": "table",
        "values": [{"x": centers[i], "y": data[i]} for i in range(h1.bin_count)]
    }]

    _add_title(h1, vega, kwargs)
    _create_scales(h1, vega, kwargs)
    _create_axes(h1, vega, kwargs)

    return vega


def _create_figure(kwargs):
    """Create basic dictionary object with figure properties."""
    return {
        "$schema": "https://vega.github.io/schema/vega/v3.json",
        "width": kwargs.pop("width", DEFAULT_WIDTH),
        "height": kwargs.pop("height", DEFAULT_HEIGHT),
        "padding": kwargs.pop("padding", DEFAULT_PADDING)
    }


def _create_scales(hist, vega, kwargs):
    """Find proper scales for axes.

    Parameters
    ----------
    hist: physt.histogram_base.HistogramBase
    vega : dict
    kwargs : dict
    """
    if hist.ndim == 1:
        bins0 = hist.bins
    else:
        bins0 = hist.bins[0]

    vega["scales"] = [
        {
            "name": "xscale",
            "type": "linear",
            "range": "width",
            "nice": True,
            "zero": None,
            "domain": [bins0[0, 0], bins0[-1, 1]],
            # "domain": {"data": "table", "field": "x"}
        },
        {
            "name": "yscale",
            "type": "linear",
            "range": "height",
            "nice": True,
            "zero": True if hist.ndim == 1 else None,
            "domain": {"data": "table", "field": "y"}
        }
    ]

    if hist.ndim >= 2:
        bins1 = hist.bins[1]
        vega["scales"][1]["domain"] = [bins1[0, 0], bins1[-1, 1]]


def _create_axes(hist, vega, kwargs):
    """

    Parameters
    ----------
    hist : physt.histogram_base.HistogramBase
        Dimensionality of histogram for which it is applicable
    vega : dict
    kwargs : dict
    """
    xlabel = kwargs.pop("xlabel", hist.axis_names[0])
    ylabel = kwargs.pop("ylabel", hist.axis_names[1] if len(hist.axis_names) == 2 else None)
    vega["axes"] = [
        {"orient": "bottom", "scale": "xscale", "title": xlabel},
        {"orient": "left", "scale": "yscale", "title": ylabel}
    ]


def _add_title(hist, vega, kwargs):
    """

    Parameters
    ----------
    hist : physt.histogram_base.HistogramBase
        Dimensionality of histogram for which it is applicable
    vega : dict
    kwargs : dict
    """
    title = kwargs.pop("title", hist.title)
    if title:
        vega["title"] = {
            "text": title
        }
