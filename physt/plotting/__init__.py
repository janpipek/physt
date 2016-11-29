"""Plotting for physt histograms.

Available backends:
- matplotlib
- bokeh
"""

from __future__ import absolute_import
from collections import OrderedDict


backends = OrderedDict()

# Use variant without exception catching if you want to debug import of backends.
# from . import matplotlib as mpl_backend
# backends["matplotlib"] = mpl_backend
# from . import bokeh as bokeh_backend
# backends["bokeh"] = bokeh_backend

try:
    from . import matplotlib as mpl_backend
    backends["matplotlib"] = mpl_backend
except:
    pass

try:
    from . import bokeh as bokeh_backend
    backends["bokeh"] = bokeh_backend
except:
    pass


if backends:
    default_backend = list(backends.keys())[0]
else:
    default_backend = None


def _get_backend(name=None):
    """Get a plotting backend.

    Tries to get it using the name - or the default one.

    Parameters
    ----------
    name: Optional[str]
        Name of the backend. If not specified, default one is selected.
    """
    if not backends:
        raise RuntimeError("No plotting backend available. Please, install matplotlib (preferred) or bokeh (limited).")
    if not name:
        name = default_backend
    backend = backends.get(name)
    if not backend:
        raise RuntimeError("Backend {0} does not exist. Use one of the following: {1}".format(name, ", ".join(backends.keys())))
    return name, backends[name]


def plot(histogram, kind=None, backend=None, **kwargs):
    """Universal plotting function.

    All keyword arguments are passed to the plotting methods.

    Parameters
    ----------
    histogram: physt.HistogramBase
    kind: Optional[str]
        Type of the plot (like "scatter", "line", ...), similar to pandas
    backend: Optional[str]
    """
    backend_name, backend = _get_backend(backend)
    if kind is None:
        kinds = [t for t in backend.types if histogram.ndim in backend.dims[t]]
        if not kinds:
            raise RuntimeError("No histogram type is supported for {0}"
                               .format(histogram.__class__.__name__))
        kind = kinds[0]
    if kind in backend.types:
        method = getattr(backend, kind)
        return method(histogram, **kwargs)
    else:
        raise RuntimeError("Histogram type error: {0} missing in backend {1}"
                           .format(kind, backend_name))


class PlottingProxy(object):
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

    def __init__(self, h):
        self.histogram = h

    def __call__(self, kind=None, **kwargs):
        """Use the plotter as callable.

        Parameters
        ----------
        histtype: Optional[str]
        """
        return plot(self.histogram, kind=kind, **kwargs)

    def __getattr__(self, name):
        """Use the plotter as a proxy object with separate plotting methods."""
        def plot_function(**kwargs):
            return plot(self.histogram, name, **kwargs)
        return plot_function

    def __dir__(self):
        _, backend = _get_backend()
        return tuple((t for t in backend.types if self.histogram.ndim in backend.dims[t]))
