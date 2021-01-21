"""Vega3 backend for plotting in physt.

The JSON can be produced without any external dependency, the ability
to show plots in-line in IPython requires 'vega3' library.

Implementation note: Values passed to JSON cannot be of type np.int64 (solution: explicit cast to float)

Common parameters
-----------------
See the `enable_inline_view` wrapper.

"""
# TODO: Custom JSON serializer better than conversion?


import codecs
import json
from functools import wraps
from typing import Any, Optional, Union, Dict

import numpy as np

from physt.histogram_collection import HistogramCollection
from physt.histogram_base import HistogramBase
from physt.histogram1d import Histogram1D
from physt.histogram_nd import Histogram2D, HistogramND
from physt.plotting.common import get_data, get_value_format, check_ndim

VEGA_IPYTHON_PLUGIN_ENABLED = False
VEGA_ERROR = None

# Options
DEFAULT_WIDTH = 400
DEFAULT_HEIGHT = 200
DEFAULT_PADDING = 5

DEFAULT_FONTSIZE = 16

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
    "YellowOrangeRed",
]
DEFAULT_PALETTE = PALETTES[0]


# Hack to find whether we can display inline images in IPython notebook
try:
    from IPython import get_ipython

    if get_ipython():
        try:
            import vega3
            from vega3 import Vega

            VEGA_IPYTHON_PLUGIN_ENABLED = True
        except ImportError:
            VEGA_ERROR = "Library 'vega3' not present"
            Vega = dict  # Convert to itself

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
        Path to write vega JSON/HTML to.
    write_format: "auto" | "json" | "html"
        Whether to create a JSON data file or a full-fledged HTML page.
    display: "auto" | True | False
        Whether to try in-line display in IPython
    indent: int
        Indentation of JSON
    """

    @wraps(f)
    def wrapper(hist, write_to=None, write_format="auto", display="auto", indent=2, **kwargs):

        vega_data = f(hist, **kwargs)

        if display is True and not VEGA_IPYTHON_PLUGIN_ENABLED:
            raise RuntimeError("Cannot display vega plot: {0}".format(VEGA_ERROR))

        if display == "auto":
            display = write_to is None

        if write_to:
            write_vega(
                vega_data,
                title=hist.title,
                write_to=write_to,
                write_format=write_format,
                indent=indent,
            )

        return display_vega(vega_data, display)

    return wrapper


def write_vega(
    vega_data, *, title: Optional[str], write_to: str, write_format: str = "auto", indent: int = 2
):
    """Write vega dictionary to an external file.

    Parameters
    ----------
    vega_data : Valid vega data as dictionary
    write_to: Path to write vega JSON/HTML to.
    write_format: "auto" | "json" | "html"
        Whether to create a JSON data file or a full-fledged HTML page.
    indent: Indentation of JSON
    """
    spec = json.dumps(vega_data, indent=indent)
    if write_format == "html" or write_format == "auto" and write_to.endswith(".html"):
        output = HTML_TEMPLATE.replace("{{ title }}", title or "Histogram").replace(
            "{{ spec }}", spec
        )
    elif write_format == "json" or write_format == "auto" and write_to.endswith(".json"):
        output = spec
    else:
        raise RuntimeError("Format not understood.")
    with codecs.open(write_to, "w", encoding="utf-8") as out:
        out.write(output)


def display_vega(vega_data: dict, display: bool = True) -> Union["Vega", dict]:
    """Optionally display vega dictionary.

    Parameters
    ----------
    vega_data : Valid vega data as dictionary
    display: Whether to try in-line display in IPython
    """
    if VEGA_IPYTHON_PLUGIN_ENABLED and display:
        return Vega(vega_data)
    else:
        return vega_data


@enable_inline_view
@check_ndim(1)
def bar(h1: "Histogram1D", **kwargs) -> dict:  # pylint: disable=blacklisted-name
    """Bar plot of 1D histogram.

    Parameters
    ----------
    lw : float
        Width of the line between bars
    alpha : float
        Opacity of the bars
    hover_alpha: float
        Opacity of the bars when hover on
    """
    # TODO: Enable collections
    # TODO: Enable legend
    if h1.ndim > 1:
        raise

    vega = _create_figure(kwargs)
    _add_title(h1, vega, kwargs)
    _create_scales(h1, vega, kwargs)
    _create_axes(h1, vega, kwargs)

    data = get_data(h1, kwargs.pop("density", None), kwargs.pop("cumulative", None)).tolist()
    lefts = h1.bin_left_edges.astype(float).tolist()
    rights = h1.bin_right_edges.astype(float).tolist()

    vega["data"] = [
        {
            "name": "table",
            "values": [
                {
                    "x": lefts[i],
                    "x2": rights[i],
                    "y": data[i],
                }
                for i in range(h1.bin_count)
            ],
        }
    ]

    alpha = kwargs.pop("alpha", 1)
    # hover_alpha = kwargs.pop("hover_alpha", alpha)

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
                    "strokeWidth": {"value": kwargs.pop("lw", 2)},
                },
                "update": {
                    "fillOpacity": [
                        # {"test": "datum === tooltip", "value": hover_alpha},
                        {"value": alpha}
                    ]
                },
            },
        }
    ]
    _create_tooltips(h1, vega, kwargs)

    return vega


DEFAULT_SCATTER_SHAPE = "circle"
# DEFAULT_SCATTER_SIZE = 2


@enable_inline_view
@check_ndim(1)
def scatter(h1: Histogram1D, **kwargs) -> dict:
    """Scatter plot of 1D histogram values.

    Points are horizontally placed in bin centers.

    Parameters
    ----------
    shape : str
    """
    shape = kwargs.pop("shape", DEFAULT_SCATTER_SHAPE)
    # size = kwargs.pop("size", DEFAULT_SCATTER_SIZE)

    mark_template = [
        {
            "type": "symbol",
            "from": {"data": "series"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    "shape": {"value": shape},
                    # "size": {"value": size},
                    "fill": {"scale": "series", "field": "c"},
                },
            },
        }
    ]
    vega = _scatter_or_line(h1, mark_template=mark_template, kwargs=kwargs)
    return vega


DEFAULT_STROKE_WIDTH = 2


@enable_inline_view
@check_ndim(1)
def line(h1: Histogram1D, **kwargs) -> dict:
    """Line plot of 1D histogram values.

    Points are horizontally placed in bin centers.

    Parameters
    ----------
    h1 : physt.histogram1d.Histogram1D
        Dimensionality of histogram for which it is applicable
    """

    lw = kwargs.pop("lw", DEFAULT_STROKE_WIDTH)

    mark_template = [
        {
            "type": "line",
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    "stroke": {"scale": "series", "field": "c"},
                    "strokeWidth": {"value": lw},
                }
            },
            "from": {"data": "series"},
        }
    ]
    vega = _scatter_or_line(h1, mark_template=mark_template, kwargs=kwargs)
    return vega


@enable_inline_view
@check_ndim(2)
def map(h2: "Histogram2D", *, show_zero: bool = True, show_values: bool = False, **kwargs) -> dict:
    """Heat-map of two-dimensional histogram."""
    vega = _create_figure(kwargs)

    values_arr = get_data(h2, kwargs.pop("density", None), kwargs.pop("cumulative", None))
    values = values_arr.tolist()
    value_format = get_value_format(kwargs.pop("value_format", None))

    _add_title(h2, vega, kwargs)
    _create_scales(h2, vega, kwargs)
    _create_axes(h2, vega, kwargs)
    _create_cmap_scale(values_arr, vega, kwargs)
    _create_colorbar(vega, kwargs)

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
            item: Dict[str, Any] = {
                "x": float(x[i]),
                "x1": float(x1[i]),
                "x2": float(x2[i]),
                "y": float(y[j]),
                "y1": float(y1[j]),
                "y2": float(y2[j]),
                "c": float(values[i][j]),
            }
            if show_values:
                item["label"] = value_format(values[i][j])
            data.append(item)

    vega["data"] = [{"name": "table", "values": data}]

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
                "update": {"fillOpacity": {"value": kwargs.pop("alpha", 1)}},
                # "hover": {
                #     "fillOpacity": {"value": 0.5}
                # }
            },
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
                        "text": {"field": "label"},
                        "x": {"scale": "xscale", "field": "x"},
                        "y": {"scale": "yscale", "field": "y"},
                    }
                },
            }
        )

    return vega


@enable_inline_view
@check_ndim(3)
def map_with_slider(
    h3: "HistogramND", *, show_zero: bool = True, show_values: bool = False, **kwargs
) -> dict:
    """Heatmap showing slice in first two dimensions, third dimension represented as a slider.

    Parameters
    ----------
    """
    vega = _create_figure(kwargs)

    values_arr = get_data(h3, kwargs.pop("density", None), kwargs.pop("cumulative", None))
    values = values_arr.tolist()
    value_format = get_value_format(kwargs.pop("value_format", None))

    _add_title(h3, vega, kwargs)
    _create_scales(h3, vega, kwargs)
    _create_axes(h3, vega, kwargs)
    _create_cmap_scale(values_arr, vega, kwargs)
    _create_colorbar(vega, kwargs)

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
                item: Dict[str, Any] = {
                    "x": float(x[i]),
                    "x1": float(x1[i]),
                    "x2": float(x2[i]),
                    "y": float(y[j]),
                    "y1": float(y1[j]),
                    "y2": float(y2[j]),
                    "k": k,
                    "c": float(values[i][j][k]),
                }
                if show_values:
                    item["label"] = value_format(values[i][j][k])
                data.append(item)

    vega["signals"] = [
        {
            "name": "k",
            "value": h3.shape[2] // 2,
            "bind": {
                "input": "range",
                "min": 0,
                "max": h3.shape[2] - 1,
                "step": 1,
                "name": (h3.axis_names[2] or "axis2") + " [slice]",
            },
        }
    ]

    vega["data"] = [
        {
            "name": "table",
            "values": data,
            "transform": [
                {
                    "type": "filter",
                    "expr": "k == datum.k",
                }
            ],
        }
    ]

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
            },
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
                        "text": {"field": "label"},
                        "x": {"scale": "xscale", "field": "x"},
                        "y": {"scale": "yscale", "field": "y"},
                    }
                },
            }
        )

    return vega


def _scatter_or_line(h1: Histogram1D, mark_template: list, kwargs: dict) -> dict:
    """Create shared properties for scatter / line plot."""
    if isinstance(h1, HistogramCollection):
        collection = h1
        h1 = h1[0]
    else:
        collection = HistogramCollection(h1)

    vega = _create_figure(kwargs)

    legend = kwargs.pop("legend", len(collection) > 1)

    vega["data"] = [
        {"name": "table", "values": []},
        {"name": "labels", "values": [h.name for h in collection]},
    ]

    for hist_i, histogram in enumerate(collection):
        centers = histogram.bin_centers.tolist()
        data = get_data(
            histogram, kwargs.pop("density", None), kwargs.pop("cumulative", None)
        ).tolist()
        vega["data"][0]["values"] += [
            {"x": centers[i], "y": data[i], "c": hist_i} for i in range(histogram.bin_count)
        ]

    _add_title(collection, vega, kwargs)
    _create_scales(collection, vega, kwargs)
    _create_axes(collection, vega, kwargs)
    _create_series_scales(vega)
    _create_series_faceted_marks(vega, mark_template)
    _create_tooltips(h1, vega, kwargs)  # TODO: Make it work!
    if legend:
        _create_series_legend(vega)

    return vega


def _create_figure(kwargs: Dict[str, Any]) -> dict:
    """Create basic dictionary object with figure properties."""
    return {
        "$schema": "https://vega.github.io/schema/vega/v3.json",
        "width": kwargs.pop("width", DEFAULT_WIDTH),
        "height": kwargs.pop("height", DEFAULT_HEIGHT),
        "padding": kwargs.pop("padding", DEFAULT_PADDING),
    }


def _create_colorbar(vega: dict, kwargs: dict):
    if kwargs.pop("show_colorbar", True):
        vega["legends"] = [{"fill": "color", "type": "gradient"}]


def _create_scales(hist: Union[HistogramCollection, HistogramBase], vega: dict, kwargs: dict):
    """Find proper scales for axes."""
    if hist.ndim == 1:
        bins0 = hist.bins.astype(float)
    else:
        bins0 = hist.bins[0].astype(float)

    xlim = kwargs.pop("xlim", "auto")
    ylim = kwargs.pop("ylim", "auto")

    nice_x = xlim == "auto"
    nice_y = ylim == "auto"

    # TODO: Unify xlim & ylim parameters with matplotlib
    # TODO: Apply xscale & yscale parameters

    vega["scales"] = [
        {
            "name": "xscale",
            "type": "linear",
            "range": "width",
            "nice": nice_x,
            "zero": None,
            "domain": [bins0[0, 0], bins0[-1, 1]]
            if xlim == "auto"
            else [float(xlim[0]), float(xlim[1])],
            # "domain": {"data": "table", "field": "x"}
        },
        {
            "name": "yscale",
            "type": "linear",
            "range": "height",
            "nice": nice_y,
            "zero": True if hist.ndim == 1 else None,
            "domain": {"data": "table", "field": "y"}
            if ylim == "auto"
            else [float(ylim[0]), float(ylim[1])],
        },
    ]

    if hist.ndim >= 2:
        bins1 = hist.bins[1].astype(float)
        vega["scales"][1]["domain"] = [bins1[0, 0], bins1[-1, 1]]


def _create_series_scales(vega: dict):
    vega["scales"].append(
        {
            "name": "series",
            "type": "ordinal",
            "range": "category",
            "domain": {"data": "table", "field": "c"},
        }
    )
    vega["scales"].append(
        {
            "name": "labels",
            "type": "ordinal",
            "range": "category",
            "domain": {"data": "labels", "field": "data"},
        }
    )


def _create_series_faceted_marks(vega: dict, pattern: list) -> None:
    vega["marks"] = [
        {
            "type": "group",
            "from": {"facet": {"name": "series", "data": "table", "groupby": "c"}},
            "marks": pattern,
        }
    ]


def _create_series_legend(vega: dict) -> None:
    vega["legends"] = [
        {
            "type": "symbol",
            "fill": "labels",
        }
    ]


def _create_cmap_scale(values_arr: np.ndarray, vega: dict, kwargs: dict):
    cmap = kwargs.pop("cmap", DEFAULT_PALETTE)
    cmap_min = float(kwargs.pop("cmap_min", values_arr.min()))
    cmap_max = float(kwargs.pop("cmap_max", values_arr.max()))

    # TODO: Apply cmap_normalize parameter

    vega["scales"].append(
        {
            "name": "color",
            "type": "sequential",
            "domain": [cmap_min, cmap_max],
            "range": {"scheme": cmap},
            "zero": False,
            "nice": False,
        }
    )


def _create_axes(hist: Union[HistogramCollection, HistogramBase], vega: dict, kwargs: dict):
    """Create axes in the figure."""
    xlabel = kwargs.pop("xlabel", hist.axis_names[0])
    ylabel = kwargs.pop("ylabel", hist.axis_names[1] if len(hist.axis_names) >= 2 else None)
    vega["axes"] = [
        {"orient": "bottom", "scale": "xscale", "title": xlabel},
        {"orient": "left", "scale": "yscale", "title": ylabel},
    ]


def _create_tooltips(hist: Histogram1D, vega: dict, kwargs: dict):
    """In one-dimensional plots, show values above the value on hover."""
    if kwargs.pop("tooltips", False):
        vega["signals"] = vega.get("signals", [])
        vega["signals"].append(
            {
                "name": "tooltip",
                "value": {},
                "on": [
                    {"events": "rect:mouseover", "update": "datum"},
                    {"events": "rect:mouseout", "update": "{}"},
                ],
            }
        )

        font_size = kwargs.get("fontsize", DEFAULT_FONTSIZE)

        vega["marks"] = vega.get("marks", [])
        vega["marks"].append(
            {
                "type": "text",
                "encode": {
                    "enter": {
                        "align": {"value": "center"},
                        "baseline": {"value": "bottom"},
                        "fill": {"value": "#333"},
                        "fontSize": {"value": font_size},
                    },
                    "update": {
                        "x": {
                            "scale": "xscale",
                            "signal": "(tooltip.x + tooltip.x2) / 2",
                            "band": 0.5,
                        },
                        "y": {"scale": "yscale", "signal": "tooltip.y", "offset": -2},
                        "text": {"signal": "tooltip.y"},
                        "fillOpacity": [
                            {"test": "datum === tooltip", "value": 0},
                            {"value": 1},
                        ],
                    },
                },
            }
        )


def _add_title(hist: Union[HistogramBase, HistogramCollection], vega: dict, kwargs: dict):
    """Display plot title if available."""
    title = kwargs.pop("title", hist.title)
    if title:
        vega["title"] = {"text": title}


HTML_TEMPLATE = """
<html>
<head>
    <title>{{ title }}</title>
        <script src="https://cdn.jsdelivr.net/npm/vega@4.2.0"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@2.6.0"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@3.18.1"></script>
    </head> 
<script>
    function render(spec) {
            view = new vega.View(vega.parse(spec))
                .renderer('canvas')  // set renderer (canvas or svg)
                .initialize('#it') // initialize view within parent DOM container
                .hover()             // enable hover encode set processing
                .run();
        }
</script>
<body>
</body>
    <div id="it"></div>
    <script>
        var spec = {{ spec }};
        render(spec);
    </script>
</html>
"""
