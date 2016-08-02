import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .common import get_data


types = ("bar", "scatter", "line", "map", "bar3d", "image")

dims = {
    "bar": [1],
    "scatter": [1],
    "line": [1],
    "map": [2],
    "bar3d": [2],
    "image": [2]
}


def bar(h1, **kwargs):
    fig, ax = get_axes(kwargs)

    add_labels(h1, ax)
    return ax


def scatter(h1, **kwargs):
    fig, ax = get_axes(kwargs)

    add_labels(h1, ax)
    return ax



def line(h1, **kwargs):
    fig, ax = get_axes(kwargs)

    add_labels(h1, ax)
    return ax


def map(h2, **kwargs):
    fig, ax = get_axes(kwargs)

    add_labels(h2, ax)
    return ax


def bar3d(h2, **kwargs):
    fig, ax = get_axes(kwargs, use_3d=True)

    density = kwargs.pop("density", False)

    data = get_data(h2, density=density, cumulative=False, flatten=True)

    colors = None
    if "color" in kwargs:
        colors = kwargs.pop("color")
    
    if "cmap" in kwargs or colors == "frequency":
        cmap = get_cmap(kwargs)
        _, cmap_data = get_cmap_data(data, kwargs)
        colors = cmap(cmap_data)
    else:
        colors = "blue"
    
    xpos, ypos = (arr.flatten() for arr in h2.get_bin_centers())
    zpos = np.zeros_like(ypos)
    dx, dy = (arr.flatten() for arr in h2.get_bin_widths())

    ax.bar3d(xpos, ypos, zpos, dx, dy, data, color=colors, **kwargs)
    ax.set_zlabel("density" if density else "frequency")

    add_labels(h2, ax)
    return ax


def image(h2, **kwargs):
    # ! Fatto
    fig, ax = get_axes(kwargs)
    cmap = get_cmap(kwargs)   # h2 as well?
    data = get_data(h2, density=kwargs.pop("density", False), cumulative=False)
    _, cmap_data = get_cmap_data(data, kwargs)

    if not "interpolation" in kwargs:
        kwargs["interpolation"] = "nearest"

    ax.imshow(cmap_data[::-1,:], cmap=cmap,
        extent=(h2.bins[0][0,0], h2.bins[0][-1,1], h2.bins[1][0,0], h2.bins[1][-1,1]),
        aspect="auto", **kwargs)

    add_labels(h2, ax)
    return ax


def get_axes(kwargs, use_3d=False):
    figsize = kwargs.pop("figsize", None)
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
        fig = ax.get_figure()
    elif use_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def get_cmap(kwargs):
    cmap = kwargs.pop("cmap", "Greys")
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    return cmap


def get_cmap_data(data, kwargs):
    transform = kwargs.pop("transform", lambda x: x)

    data = np.asarray(transform(data))
  
    cmap_max = kwargs.pop("cmap_max", data.max())
    cmap_min = kwargs.pop("cmap_min", 0)
    if cmap_min == "min":
        cmap_min = data.min()
    norm = colors.Normalize(cmap_min, cmap_max, clip=True)
    return norm, norm(data)


def add_labels(h, ax):
    if h.name:
        ax.set_title(h.name)
    if hasattr(h, "axis_name") and h.axis_name:
        ax.set_xlabel(h.axis_name)
    else:
        if h.axis_names[0]:
            ax.set_xlabel(h.axis_names[0])
        if h.axis_names[1]:
            ax.set_ylabel(h.axis_names[1])
    ax.get_figure().tight_layout()


def add_colorbar(fig, cmap, cmap_data, norm):
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(cmap_data)   # TODO: Or what???

    fig.colorbar(mappable, ax=ax)            
