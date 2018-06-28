"""Plot.ly backend for plotting in physt.

Currently, it uses matplotlib translation for 1D histograms:
- bar
- scatter
- line

TODO: More elaborate output planned
"""
from functools import wraps

from . import matplotlib as mpl_backend


def _wrap_matplotlib_f(f):
    import plotly.offline as pyo
    import plotly.plotly as pyp

    @wraps(f)
    def new_f(*args, offline=True, filename=None, **kwargs):
        py = pyo if offline else pyp
        ax = f(*args, **kwargs)
        fig = ax.figure
        if filename:
            return py.plot_mpl(fig, filename=filename)
        else:
            return py.iplot_mpl(fig)

    return new_f


types = ["bar", "scatter", "line"]
dims = {x : [1]  for x in types}

for plot_type in types:
    globals()[plot_type] = _wrap_matplotlib_f(mpl_backend.__dict__[plot_type])
    
    
del wraps, _wrap_matplotlib_f, mpl_backend, plot_type
