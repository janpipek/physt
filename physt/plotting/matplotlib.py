import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .common import get_data, transform_data


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


def map(h2, show_zero=True, show_values=False, show_colorbar=None, **kwargs):
    fig, ax = get_axes(kwargs)

    format_value = kwargs.pop("format_value", lambda x: x)
    transform = kwargs.get("transform", False)

    if show_colorbar is None:
        show_colorbar = not transform


    data = get_data(h2, cumulative=False, flatten=True, density=kwargs.pop("density", False))    
    transformed = transform_data(data, kwargs)
    

    cmap = get_cmap(kwargs)
    norm, cmap_data = get_cmap_data(transformed, kwargs)
    colors = cmap(cmap_data)    

    xpos, ypos = (arr.flatten() for arr in h2.get_bin_left_edges())
    dx, dy = (arr.flatten() for arr in h2.get_bin_widths())
    text_x, text_y = (arr.flatten() for arr in h2.get_bin_centers())

    xlim = kwargs.get("xlim", (h2.bins[0][0,0], h2.bins[0][-1,1]))
    ylim = kwargs.get("ylim", (h2.bins[1][0,0], h2.bins[1][-1,1]))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.autoscale_view() 

    alphas = get_alpha_data(cmap_data, kwargs)
    if np.isscalar(alphas):
        alphas = np.ones_like(data) * alphas

    for i in range(len(xpos)):
        bin_color = colors[i]
        alpha = alphas[i]

        if data[i] != 0 or show_zero:
            rect = plt.Rectangle([xpos[i], ypos[i]], dx[i], dy[i],
                facecolor=bin_color, edgecolor=kwargs.get("grid_color", cmap(0.5)),
                lw=kwargs.get("lw", 0.5), alpha=alpha)
            ax.add_patch(rect)

            if show_values:
                text = format_value(data[i])
                yiq_y = np.dot(bin_color[:3], [0.299, 0.587, 0.114])
                    
                if yiq_y > 0.5:
                    text_color = (0.0, 0.0, 0.0, kwargs.get("text_alpha", alpha))
                else:
                    text_color = (1.0, 1.0, 1.0, kwargs.get("text_alpha", alpha))
                ax.text(text_x[i], text_y[i], text, horizontalalignment='center', verticalalignment='center', color=text_color, clip_on=True)              

    if show_colorbar:
        if transform:
            raise RuntimeError("Cannot plot colorbar with transformed values.")
        mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(cmap_data)
        fig.colorbar(mappable, ax=ax)

    add_labels(h2, ax)
    return ax


def bar3d(h2, **kwargs):
    fig, ax = get_axes(kwargs, use_3d=True)
    density = kwargs.pop("density", False)
    data = get_data(h2, cumulative=False, flatten=True, density=density)
    transformed = transform_data(data, kwargs)
    
    if "cmap" in kwargs:
        cmap = get_cmap(kwargs)
        _, cmap_data = get_cmap_data(transformed, kwargs)
        colors = cmap(cmap_data)
    else:
        colors = kwargs.pop("color", "blue")
    
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
    data = get_data(h2, cumulative=False, density=kwargs.pop("density", False))
    transformed = transform_data(data, kwargs)
    _, cmap_data = get_cmap_data(transformed, kwargs)

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
    cmap_max = kwargs.pop("cmap_max", data.max())
    cmap_min = kwargs.pop("cmap_min", 0)
    if cmap_min == "min":
        cmap_min = data.min()
    norm = colors.Normalize(cmap_min, cmap_max, clip=True)
    return norm, norm(data)


def get_alpha_data(data, kwargs):
    alpha = kwargs.pop("alpha", 1)
    if hasattr(alpha, "__call__"):
        return np.vectorize(alpha)(data)
    return alpha


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
