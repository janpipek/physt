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

types = ("bar", "scatter", "line", "map")

dims = {
    "bar": [1],
    "scatter": [1],
    "line": [1],
    "map": [2]
}


def enable_inline_view(f):
    @wraps(f)
    def wrapper(hist, write_to=None, display="auto", *args, **kwargs):
        vega_data = f(hist, *args, **kwargs)

        if display is True and not VEGA_IPYTHON_PLUGIN_ENABLED:
            raise RuntimeError("Cannot display vega plot: {0}".format(VEGA_ERROR))

        if display == "auto":
            display = write_to is None

        if write_to is not None:
            with codecs.open(write_to, "w", encoding="utf-8") as out:
                json.dump(vega_data, out)

        if VEGA_IPYTHON_PLUGIN_ENABLED and display:
            from vega3 import Vega
            return Vega(vega_data)
        else:
            return vega_data

    return wrapper


@enable_inline_view
def bar(h1, **kwargs):
    """

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
    """

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
def map(h2, **kwargs):
    vega = _create_figure(kwargs)
    return vega


def _scatter_or_line(h1, kwargs):
    """

    Parameters
    ----------
    h1 : physt.histogram1d.Histogram1D
        Dimensionality of histogram for which it is applicable
    vega : dict
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
    return {
        "$schema": "https://vega.github.io/schema/vega/v3.json",
        "width": kwargs.pop("width", DEFAULT_WIDTH),
        "height": kwargs.pop("height", DEFAULT_HEIGHT),
        "padding": kwargs.pop("padding", DEFAULT_PADDING)
    }


def _create_scales(hist, vega, kwargs):
    """

    Parameters
    ----------
    vega : dict
    kwargs : dict
    """
    vega["scales"] = [
        {
            "name": "xscale",
            "type": "linear",
            "range": "width",
            "nice": True,
            "domain": {"data": "table", "field": "x"}
        },
        {
            "name": "yscale",
            "type": "linear",
            "range": "height",
            "nice": True,
            "zero": True,
            "domain": {"data": "table", "field": "y"}
        }
    ]


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
