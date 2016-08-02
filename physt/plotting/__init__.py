from collections import OrderedDict


backends = OrderedDict()

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

default_backend = list(backends.keys())[0]


def _get_backend(kwargs={}):
    name = kwargs.pop("backend", default_backend)
    return name, backends[name]


def plot(histogram, histtype=None, **kwargs):
    backend_name, backend = _get_backend(kwargs)
    if histtype is None:
        histtype = [t for t in backend.types if histogram.ndim in backend.dims[t]][0]
    if histtype in backend.types:
        method = getattr(backend, histtype)
        return method(histogram, **kwargs)
    else:
        raise RuntimeError("Histogram type error: {0} missing in backend {1}".format(histtype, backend_name))


class Plotter(object):
    def __init__(self, h):
        self.h = h

    def __call__(self, histtype=None, **kwargs):
        return plot(self.h, histtype, **kwargs)

    def __getattr__(self, name):
        def f(**kwargs):
            return plot(self.h, name, **kwargs)
        return f

    def __dir__(self):
        _, backend = _get_backend()
        return tuple((t for t in backend.types if self.h.ndim in backend.dims[t]))
