import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .common import get_data, transform_data, get_err_data


types = ("bar", "scatter", "line", "map", "bar3d", "image")

dims = {
    "bar": [1],
    "scatter": [1],
    "line": [1],
    "map": [2],
    "bar3d": [2],
    "image": [2]
}


def bar(h1, errors=False, **kwargs):
    fig, ax = get_axes(kwargs)

    stats_box = kwargs.pop("stats_box", False)
    show_values = kwargs.pop("show_values", False)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)

    data = get_data(h1, cumulative=cumulative, density=density) 
    transformed = transform_data(data, kwargs)

    if "cmap" in kwargs:
        cmap = get_cmap(kwargs)
        _, cmap_data = get_cmap_data(transformed, kwargs)
        colors = cmap(cmap_data)
    else:
        colors = kwargs.pop("color", "blue")    

    apply_xy_lims(ax, h1, data, kwargs)
    add_ticks(ax, h1, kwargs)
    
    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        kwargs["yerr"] = err_data
        if not "ecolor" in kwargs:
            kwargs["ecolor"] = "black"        

    ax.bar(h1.bin_left_edges, data, h1.bin_widths, color=colors, **kwargs)
    add_labels(h1, ax)

    if show_values:
        add_values(ax, h1, data)
    if stats_box:
        add_stats_box(h1, ax)

    return ax


def scatter(h1, errors=False, **kwargs):
    fig, ax = get_axes(kwargs)

    stats_box = kwargs.pop("stats_box", False)
    show_values = kwargs.pop("show_values", False)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)

    data = get_data(h1, cumulative=cumulative, density=density)
    transformed = transform_data(data, kwargs)

    if "cmap" in kwargs:
        cmap = get_cmap(kwargs)
        _, cmap_data = get_cmap_data(transformed, kwargs)
        kwargs["color"] = cmap(cmap_data)
    else:
        kwargs["color"] = kwargs.pop("color", "blue")   

    apply_xy_lims(ax, h1, data, kwargs)
    add_ticks(ax, h1, kwargs)    

    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        ax.errorbar(h1.bin_centers, data, yerr=err_data, fmt=kwargs.get("fmt", "o"), ecolor=kwargs.get("ecolor", "black"))
    else:
        ax.scatter(h1.bin_centers, data, **kwargs)

    add_labels(h1, ax)

    if show_values:
        add_values(ax, h1, data)
    if stats_box:
        add_stats_box(h1, ax)    
    return ax



def line(h1, errors=False, **kwargs):
    fig, ax = get_axes(kwargs)

    stats_box = kwargs.pop("stats_box", False)
    show_values = kwargs.pop("show_values", False)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)

    data = get_data(h1, cumulative=cumulative, density=density) 
    apply_xy_lims(ax, h1, data, kwargs)
    add_ticks(ax, h1, kwargs)   

    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        ax.errorbar(h1.bin_centers, data, yerr=err_data, fmt=kwargs.get("fmt", "-"), ecolor=kwargs.get("ecolor", "black"), **kwargs)
    else:
        ax.plot(h1.bin_centers, data, **kwargs)

    add_labels(h1, ax)

    if stats_box:
        add_stats_box(h1, ax)
    if show_values:
        add_values(ax, h1, data)
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
    if hasattr(h, "axis_name"):
        if h.axis_name:
            ax.set_xlabel(h.axis_name)
    else:
        if h.axis_names[0]:
            ax.set_xlabel(h.axis_names[0])
        if h.axis_names[1]:
            ax.set_ylabel(h.axis_names[1])
    ax.get_figure().tight_layout()

def add_values(ax, h1, data):
    for x, y in zip(h1.bin_centers, data):
        ax.text(x, y, str(y), ha='center', va='bottom')  


def add_colorbar(fig, cmap, cmap_data, norm):
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(cmap_data)   # TODO: Or what???

    fig.colorbar(mappable, ax=ax)     


def add_stats_box(h1, ax):
    # place a text box in upper left in axes coords
    text = "Total: {0}\nMean: {1:.2f}\nStd.dev: {2:.2f}".format(h1.total, h1.mean(), h1.std())
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')      


def apply_xy_lims(ax, h1, data, kwargs):
    xscale = kwargs.pop("xscale", None)
    yscale = kwargs.pop("yscale", None)
    ylim = kwargs.pop("ylim", "auto")
    xlim = kwargs.pop("xlim", "auto")

    if ylim is not "keep":
        if isinstance(ylim, tuple):
            pass
        else:
            ylim = ax.get_ylim()
            if data.size > 0 and data.max() > 0:
                ylim = (0, max(ylim[1], data.max() + (data.max() - ylim[0]) * 0.1))
            if yscale == "log":
                ylim = (abs(data[data > 0].min()) * 0.9, ylim[1] * 1.1)
        ax.set_ylim(ylim)

    if xlim is not "keep":
        if isinstance(xlim, tuple):
            pass
        else:
            xlim = ax.get_xlim()
            if len(h1.bin_centers) > 2:
                xlim = (h1.bin_left_edges[0], h1.bin_right_edges[-1])
            if xscale == "log":
                if xlim[0] <= 0:
                    raise RuntimeError("Cannot use logarithmic scale for non-positive bins.")
        ax.set_xlim(xlim)

    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)


def add_ticks(ax, h1, kwargs):
    ticks = kwargs.pop("ticks", None)
    if not ticks:
        return
    elif ticks == "center":
        ax.set_xticks(h1.bin_centers)
    elif ticks == "edge":
        ax.set_xticks(h1.bin_left_edges)

