"""Plotting for physt histograms.

Available backends
------------------
- matplotlib
- vega
- plotly (simple wrapper around matplotlib for 1D histograms)
- folium (just for the geographical histograms)

Calling the plotting functions

Common parameters
-----------------
There are several backends (and user-defined may be added) and several
plotting functions for each - we try to keep a consistent set of
parameters to which all implementations should try to stick (with exceptions).

All histograms
~~~~~~~~~~~~~~
write_to : str (optional)
    Path to file where the output will be stored
title : str (optional)
    String to be displayed as plot title (defaults to h.title)
xlabel : str (optional)
    String to be displayed as x-axis label (defaults to corr. axis name)
ylabel : str (optional)
    String to be displayed as y-axis label (defaults to corr. axis name)
xscale : str (optional)
    If "log", x axis will be scaled logarithmically
yscale : str (optional)
    If "log", y axis will be scaled logarithmically
xlim : tuple | "auto" | "keep"

ylim : tuple | "auto" | "keep"

invert_y : bool
    If True, the y axis points downwards
ticks : {"center", "edge"}, optional
    If set, each bin will have a tick (either central or edge)
alpha : float (optional)
    The alpha of the whole plot (default: 1)
cmap : str or list
    Name of the palette or list of colors or something that the
    respective backend can interpret as colourmap.
cmap_normalize : {"log"}, optional

cmap_min :

cmap_max :

show_values : bool
    If True, show values next to (or inside) the bins
value_format : str or Callable
    How bin values (if to be displayed) are rendered.
zorder : int (optional)

text_color :
text_alpha :
text_* :
    Other options that are passed to the formatting of values without the prefix

1D histograms
~~~~~~~~~~~~~
cumulative : bool
    If True, show CDF instead of bin heights
density : bool
    If True, does not show bin contents but contents divided by width
errors : bool
    Whether to show error bars (if available)
show_stats : bool
    If True, display a small box with statistical info

2D heatmaps
~~~~~~~~~~~
show_zero : bool
    Whether to show bins that have 0 frequency
grid_color :
    Colour of line between bins
show_colorbar : bool
    Whether to display a colorbar next to the plot itself

Line plots
~~~~~~~~~~
lw (or linewidth) : int
    Width of the lines
"""
from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from physt.types import HistogramBase, HistogramCollection

from . import ascii as ascii_backend

backends: Dict[str, Any] = {}

if TYPE_CHECKING:
    from typing import Any, Dict, Optional, Tuple, Union

# Use variant without exception catching if you want to debug import of backends.
# from . import matplotlib as mpl_backend
# backends["matplotlib"] = mpl_backend
# from . import folium as folium_backend
# backends["folium"] = folium_backend
# from . import vega as vega_backend
# backends["vega"] = vega_backend
# from . import plotly as plotly_backend
# backends["plotly"] = plotly_backend

with suppress(ImportError):
    from . import matplotlib as mpl_backend

    backends["matplotlib"] = mpl_backend


with suppress(ImportError):
    from . import vega as vega_backend

    backends["vega"] = vega_backend

with suppress(ImportError):
    from . import plotly as plotly_backend

    backends["plotly"] = plotly_backend


with suppress(ImportError):
    from . import folium as folium_backend

    backends["folium"] = folium_backend


backends["ascii"] = ascii_backend

_default_backend: Optional[str] = None

if backends:
    _default_backend = list(backends.keys())[0]


def set_default_backend(name: str) -> None:
    """Choose a default backend."""
    global _default_backend  # pylint: disable=global-statement
    if name == "bokeh":
        raise ValueError(
            "Support for bokeh has been discontinued."
            "At some point, we may return to support holoviews."
        )
    if name not in backends:
        raise ValueError(f"Backend '{name}' is not supported and cannot be set as default.")
    _default_backend = name


def get_default_backend() -> Optional[str]:
    """The backend that will be used by default with the `plot` function."""
    return _default_backend


def _get_backend(name: Optional[str] = None) -> Tuple[str, Any]:
    """Get a plotting backend.

    Tries to get it using the name - or the default one.
    """
    if not backends:
        raise RuntimeError(
            "No plotting backend available. Please, install matplotlib (preferred),"
            " plotly or vega (more limited)."
        )
    if not name:
        name = _default_backend
        if not name:
            raise RuntimeError("No backend for physt plotting.")
    if name == "bokeh":
        raise NotImplementedError(
            "Support for bokeh has been discontinued."
            " At some point, we may return to support holoviews."
        )
    backend = backends.get(name)
    if not backend:
        available = ", ".join(backends.keys())
        raise RuntimeError(f"Backend {name} does not exist. Use one of the following: {available}.")
    return name, backend


def plot(
    histogram: Union[HistogramBase, HistogramCollection],
    kind: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs,
):
    """Universal plotting function.

    All keyword arguments are passed to the plotting methods.

    Parameters
    ----------
    kind: Type of the plot (like "scatter", "line", ...), similar to pandas
    """
    backend_name, impl = _get_backend(backend)
    if kind is None:
        kinds = [t for t in impl.types if histogram.ndim in impl.dims[t]]  # type: ignore
        if not kinds:
            raise RuntimeError(f"No plot type is supported for {histogram.__class__.__name__}")
        kind = kinds[0]
    if kind in impl.types:
        method = getattr(impl, kind)
        return method(histogram, **kwargs)
    else:
        raise RuntimeError(f"Histogram type error: {kind} missing in backend {backend_name}.")


class PlottingProxy:
    """Proxy enabling to call plotting methods on histogram objects.

    It can be used both as a method or as an object containing methods. In any case,
    it only forwards the call to the universal plot() function.

    The __dir__ method should offer all plotting methods supported by the currently
    selected backend.

    Example
    -------
        plotter = histogram.plot
        plotter(...)          # Plots using defaults
        plotter.bar(...)      # Plots as a specified plot type ("bar")

    Note
    ----
    Inspiration taken from the way how pandas deals with this.

    """

    def __init__(self, h: Union[HistogramBase, HistogramCollection]):
        self.histogram = h

    def __call__(self, kind: Optional[str] = None, **kwargs):
        """Use the plotter as callable."""
        return plot(self.histogram, kind=kind, **kwargs)

    def __getattr__(self, name: str):
        """Use the plotter as a proxy object with separate plotting methods."""

        def plot_function(**kwargs):
            return plot(self.histogram, name, **kwargs)

        return plot_function

    def __dir__(self):
        _, backend = _get_backend()
        return tuple((t for t in backend.types if self.histogram.ndim in backend.dims[t]))
