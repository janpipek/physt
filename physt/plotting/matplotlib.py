"""
Matplotlib backend for plotting in physt.

This module contains several plotting functions and a lot of underscored
helper functions. User is expected to use only the former ones.

Plot functions for 1D histograms
- bar
- scatter
- line

Plot functions for 2D histograms
- map
- image
- bar3d
- polar_map

Each plotting method supports many parameters. These are quite common
and very often corresponding to a matplotlib parameter of a same name.
The very general keyword argument dict is sequentially forwarded to
plotting helper functions that do part of the plotting job and popped of the used
parameters.
"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .common import get_data, get_err_data


types = ("bar", "scatter", "line", "map", "bar3d", "image", "polar_map")

dims = {
    "bar": [1],
    "scatter": [1],
    "line": [1],
    "map": [2],
    "bar3d": [2],
    "image": [2],
    "polar_map": [2]
}


def bar(h1, errors=False, **kwargs):
    """Bar plot of 1D histograms.

    Parameters
    ----------
    h1: Histogram1D
    errors: bool
        Whether to draw error bars.

    Returns
    -------
    plt.Axes
    """
    fig, ax = _get_axes(kwargs)

    stats_box = kwargs.pop("stats_box", False)
    show_values = kwargs.pop("show_values", False)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)
    label = kwargs.pop("label", h1.name)

    data = get_data(h1, cumulative=cumulative, density=density)
    # transformed = transform_data(data, kwargs)

    if "cmap" in kwargs:
        cmap = _get_cmap(kwargs)
        _, cmap_data = _get_cmap_data(data, kwargs)
        colors = cmap(cmap_data)
    else:
        colors = kwargs.pop("color", "blue")

    _apply_xy_lims(ax, h1, data, kwargs)
    _add_ticks(ax, h1, kwargs)

    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        kwargs["yerr"] = err_data
        if not "ecolor" in kwargs:
            kwargs["ecolor"] = "black"

    ax.bar(h1.bin_left_edges, data, h1.bin_widths, label=label, color=colors, **kwargs)
    _add_labels(h1, ax)

    if show_values:
        _add_values(ax, h1, data)
    if stats_box:
        _add_stats_box(h1, ax)

    return ax


def scatter(h1, errors=False, **kwargs):
    """Scatter plot of 1D histogram.

    Parameters
    ----------
    h1: Histogram1D
    errors: bool
        Whether to draw error bars.

    Returns
    -------
    plt.Axes
    """
    fig, ax = _get_axes(kwargs)

    stats_box = kwargs.pop("stats_box", False)
    show_values = kwargs.pop("show_values", False)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)

    data = get_data(h1, cumulative=cumulative, density=density)
    # transformed = transform_data(data, kwargs)

    if "cmap" in kwargs:
        cmap = _get_cmap(kwargs)
        _, cmap_data = _get_cmap_data(data, kwargs)
        kwargs["color"] = cmap(cmap_data)
    else:
        kwargs["color"] = kwargs.pop("color", "blue")

    _apply_xy_lims(ax, h1, data, kwargs)
    _add_ticks(ax, h1, kwargs)

    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        ax.errorbar(h1.bin_centers, data, yerr=err_data, fmt=kwargs.get("fmt", "o"), ecolor=kwargs.get("ecolor", "black"))
    else:
        ax.scatter(h1.bin_centers, data, **kwargs)

    _add_labels(h1, ax)

    if show_values:
        _add_values(ax, h1, data)
    if stats_box:
        _add_stats_box(h1, ax)
    return ax


def line(h1, errors=False, **kwargs):
    """Line plot of 1D histogram.

    Parameters
    ----------
    h1 : Histogram1D
    errors : bool
        Whether to draw error bars.

    Returns
    -------
    plt.Axes
    """
    fig, ax = _get_axes(kwargs)

    stats_box = kwargs.pop("stats_box", False)
    show_values = kwargs.pop("show_values", False)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)

    data = get_data(h1, cumulative=cumulative, density=density)
    _apply_xy_lims(ax, h1, data, kwargs)
    _add_ticks(ax, h1, kwargs)

    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        ax.errorbar(h1.bin_centers, data, yerr=err_data, fmt=kwargs.get("fmt", "-"), ecolor=kwargs.get("ecolor", "black"), **kwargs)
    else:
        ax.plot(h1.bin_centers, data, **kwargs)

    _add_labels(h1, ax)

    if stats_box:
        _add_stats_box(h1, ax)
    if show_values:
        _add_values(ax, h1, data)
    return ax


def map(h2, show_zero=True, show_values=False, show_colorbar=True, **kwargs):
    """Coloured-rectangle plot of 2D histogram.

    Parameters
    ----------
    h2 : Histogram2D
    show_zero : Optional[bool]
        Whether to show coloured box for bins with 0 frequency (otherwise background).
    show_values : Optional[bool]
        Whether to show labels with frequencies/densities in the middle of the bin
    show_colorbar : Optional[bool]
        Whether to show colorbar next to the plot.
    grid_color : Optional
        Colour of line between bins
    text_color : Optional
        Colour of text descriptions
    text_alpha : Optional[float]
        Alpha for the text labels only

    Returns
    -------
    plt.Axes

    See Also
    --------
    image, polar_map

    Notes
    -----
    It is not possible to draw colorbar with transformed values.
    """
    fig, ax = _get_axes(kwargs)

    format_value = kwargs.pop("format_value", lambda x: x)

    data = get_data(h2, cumulative=False, flatten=True, density=kwargs.pop("density", False))
    # transformed = transform_data(data, kwargs)


    cmap = _get_cmap(kwargs)
    norm, cmap_data = _get_cmap_data(data, kwargs)
    colors = cmap(cmap_data)

    xpos, ypos = (arr.flatten() for arr in h2.get_bin_left_edges())
    dx, dy = (arr.flatten() for arr in h2.get_bin_widths())
    text_x, text_y = (arr.flatten() for arr in h2.get_bin_centers())

    _apply_xy_lims(ax, h2, data=data, kwargs=kwargs)
    ax.autoscale_view()

    alphas = _get_alpha_data(cmap_data, kwargs)
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

                text_color = kwargs.get("text_color", None)
                if not text_color:
                    if yiq_y > 0.5:
                        text_color = (0.0, 0.0, 0.0, kwargs.get("text_alpha", alpha))
                    else:
                        text_color = (1.0, 1.0, 1.0, kwargs.get("text_alpha", alpha))
                ax.text(text_x[i], text_y[i], text, horizontalalignment='center',
                        verticalalignment='center', color=text_color, clip_on=True)

    if show_colorbar:
        _add_colorbar(ax, cmap, cmap_data, norm)
    _add_labels(h2, ax)
    return ax


def bar3d(h2, **kwargs):
    """Plot of 2D histograms as 3D boxes.

    Parameters
    ----------
    h2 : Histogram2D

    Returns
    -------
    plt.Axes
    """
    fig, ax = _get_axes(kwargs, use_3d=True)
    density = kwargs.pop("density", False)
    data = get_data(h2, cumulative=False, flatten=True, density=density)
    # transformed = transform_data(data, kwargs)

    if "cmap" in kwargs:
        cmap = _get_cmap(kwargs)
        _, cmap_data = _get_cmap_data(data, kwargs)
        colors = cmap(cmap_data)
    else:
        colors = kwargs.pop("color", "blue")

    xpos, ypos = (arr.flatten() for arr in h2.get_bin_centers())
    zpos = np.zeros_like(ypos)
    dx, dy = (arr.flatten() for arr in h2.get_bin_widths())

    ax.bar3d(xpos, ypos, zpos, dx, dy, data, color=colors, **kwargs)
    ax.set_zlabel("density" if density else "frequency")

    _add_labels(h2, ax)
    return ax


def image(h2, show_colorbar=True, **kwargs):
    """Plot of 2D histograms based on pixmaps.

    Similar to map, but it:
    - has fewer options
    - is much more effective (enables thousands)
    - does not support irregular bins

    Parameters
    ----------
    h2: physt.histogram_nd.Histogram2D
    interpolation: str
        interpolation parameter passed to imshow, default: "nearest" (creates rectangles)

    Returns
    -------
    plt.Axes
    """
    # TODO: Check regular bins.

    fig, ax = _get_axes(kwargs)
    cmap = _get_cmap(kwargs)   # h2 as well?
    data = get_data(h2, cumulative=False, density=kwargs.pop("density", False))
    # transformed = transform_data(data, kwargs)
    norm, cmap_data = _get_cmap_data(data, kwargs)

    for binning in h2._binnings:
        if not binning.is_regular():
            raise RuntimeError("Histograms with irregular bins cannot be plotted using image method.")

    if not "interpolation" in kwargs:
        kwargs["interpolation"] = "nearest"
    if kwargs.get("xscale") == "log" or kwargs.get("yscale") == "log":
        raise RuntimeError("Cannot use logarithmic axes with image plots.")

    _apply_xy_lims(ax, h2, data=data, kwargs=kwargs)

    ax.imshow(cmap_data.T[::-1,:], cmap=cmap,
        extent=(h2.bins[0][0,0], h2.bins[0][-1,1], h2.bins[1][0,0], h2.bins[1][-1,1]),
        aspect="auto", **kwargs)

    if show_colorbar:
        _add_colorbar(ax, cmap, cmap_data, norm)
    _add_labels(h2, ax)

    return ax


def polar_map(hist, show_zero=True, **kwargs):
    """Polar map of polar histograms.

    Similar to map, but supports less parameters.

    Returns
    -------
    plt.Axes
    """
    fig, ax = _get_axes(kwargs, use_polar=True)

    data = get_data(hist, cumulative=False, flatten=True, density=kwargs.pop("density", False))
    # transformed = transform_data(data, kwargs)

    cmap = _get_cmap(kwargs)
    norm, cmap_data = _get_cmap_data(data, kwargs)
    colors = cmap(cmap_data)

    rpos, phipos = (arr.flatten() for arr in hist.get_bin_left_edges())
    dr, dphi  = (arr.flatten() for arr in hist.get_bin_widths())
    rmax, _ =  (arr.flatten() for arr in hist.get_bin_right_edges())

    alphas = _get_alpha_data(cmap_data, kwargs)
    if np.isscalar(alphas):
        alphas = np.ones_like(data) * alphas

    for i in range(len(rpos)):
        if data[i] > 0 or show_zero:
            bin_color = colors[i]
            bars = ax.bar(phipos[i], dr[i], width=dphi[i], bottom=rpos[i], color=bin_color,
                          edgecolor=kwargs.get("grid_color", cmap(0.5)), lw=kwargs.get("lw", 0.5),
                          alpha=alphas[i])

    ax.set_rmax(rmax.max())
    return ax


def pair_bars(first, second, **kwargs):
    """Draw two different histograms mirrored in one figure.

    Parameters
    ----------
    first: Histogram1D
    second: Histogram1D
    color1:
    color2:

    Returns
    -------
    plt.Axes
    """
    _, ax = _get_axes(kwargs)
    color1 = kwargs.pop("color1", "red")
    color2 = kwargs.pop("color2", "blue")
    title = kwargs.pop("title", "{0} - {1}".format(first.name, second.name))
    xlim = kwargs.pop("xlim", (min(first.bin_left_edges[0], first.bin_left_edges[0]), max(first.bin_right_edges[-1], second.bin_right_edges[-1])))

    bar(first * (-1), color=color1, ax=ax, ylim="keep", **kwargs)
    bar(second, color=color2, ax=ax, ylim="keep", **kwargs)
    ax.set_title(title)
    ticks = np.abs(ax.get_yticks())
    if np.allclose(np.rint(ticks), ticks):
        ax.set_yticklabels(ticks.astype(int))
    else:
        ax.set_yticklabels(ticks)
    ax.set_xlim(xlim)
    ax.legend()
    return ax


def _get_axes(kwargs, use_3d=False, use_polar=False):
    """Prepare the axis to draw into.

    Parameters
    ----------
    use_3d: bool
        If yes, an axis with 3D projection is created.
    use_polar: bool
        If yes, the plot will have polar coordinates.

    Kwargs
    ------
    ax: Optional[plt.Axes]
        An already existing axis to be used.
    figsize: Optional[tuple]
        Size of the new figure (if no axis is given).

    Returns
    ------
    fig : plt.Figure
    ax : plt.Axes
    """
    figsize = kwargs.pop("figsize", None)
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
        fig = ax.get_figure()
    elif use_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    elif use_polar:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _get_cmap(kwargs):
    """Get the colour map for plots that support it.

    Parameters
    ----------
    cmap : str or colors.Colormap
        A map or an instance of cmap. This can also be a seaborn palette
        (if seaborn is installed).

    Returns
    -------
    colors.Colormap
    """
    cmap = kwargs.pop("cmap", "Greys")
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except BaseException as exc:
            try:
                import seaborn.apionly as sns
                cmap = sns.color_palette(as_cmap=True)
            except ImportError:
                raise exc
    return cmap


def _get_cmap_data(data, kwargs):
    """Get normalized values to be used with a colormap.

    Parameters
    ----------
    data : array_like
    cmap_min : Optional[float] or "min"
        By default 0. If "min", minimum value of the data.
    cmap_max : Optional[float]
        By default, maximum value of the data
    cmap_normalize : str or colors.Normalize

    Returns
    -------
    normalizer : colors.Normalize
    normalized_data : array_like
    """
    norm = kwargs.pop("cmap_normalize", None)
    if norm == "log":
        cmap_max = kwargs.pop("cmap_max", data.max())
        cmap_min = kwargs.pop("cmap_min", data[data > 0].min())
        norm = colors.LogNorm(cmap_min, cmap_max)
    elif not norm:
        cmap_max = kwargs.pop("cmap_max", data.max())
        cmap_min = kwargs.pop("cmap_min", 0)
        if cmap_min == "min":
            cmap_min = data.min()
        norm = colors.Normalize(cmap_min, cmap_max, clip=True)
    return norm, norm(data)


def _get_alpha_data(data, kwargs):
    """Get alpha values for all data points.

    Parameters
    ----------
    data : array_like
    alpha: Callable or float
        This can be a fixed value or a function of the data.

    Returns
    -------
    array_like
    """
    alpha = kwargs.pop("alpha", 1)
    if hasattr(alpha, "__call__"):
        return np.vectorize(alpha)(data)
    return alpha


def _add_labels(h, ax):
    """Add axis and plot labels.

    Parameters
    ----------
    ax : plt.Axes
    h : Histogram1D or Histogram2D
    """
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


def _add_values(ax, h1, data):
    """Show values next to each bin in a 1D plot.

    Parameters
    ----------
    ax : plt.Axes
    h1 : physt.histogram1d.Histogram1D
    data : array_like
        The values to be displayed

    # TODO: Add some formatting
    """
    for x, y in zip(h1.bin_centers, data):
        ax.text(x, y, str(y), ha='center', va='bottom', clip_on=True)


def _add_colorbar(ax, cmap, cmap_data, norm):
    """Show a colorbar right of the plot.

    Parameters
    ----------
    ax : plt.Axes
    cmap : colors.Colormap
    cmap_data : array_like
    norm : colors.Normalize
    """
    fig = ax.get_figure()
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(cmap_data)   # TODO: Or what???
    fig.colorbar(mappable, ax=ax)


def _add_stats_box(h1, ax):
    """Insert a small legend-like box with statistical information.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw it into
    h1 : physt.histogram1d.Histogram1D
        Histogram with valid statistics information

    Note
    ----
    Very basic implementation.
    """

    # place a text box in upper left in axes coords
    text = "Total: {0}\nMean: {1:.2f}\nStd.dev: {2:.2f}".format(h1.total, h1.mean(), h1.std())
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')


def _apply_xy_lims(ax, h1, data, kwargs):
    """Apply axis limits and scales from kwargs.

    Parameters
    ----------
    ax : plt.Axes
    h1 : Histogram1D or Histogram2D
    data : np.ndarray
        The frequencies or densities or otherwise manipulated data
    kwargs: dict
        xscale : Optional[str]
            If "log", the horizontal axis will use logarithmic scale
        yscale : Optional[str]
            If "log", the vertical axis will use logarithmic scale
        xlim : { "keep", "auto" } or tuple(float)
            "auto" (default) - the axis will fit first and last bin edges
            "keep" - let matlotlib figure this out
            tuple - standard parameter for set_xlim
        ylim : { "keep", "auto" } or float
            "auto" (default)
                - the axis will fit first and last bin edges (2D)
                - the axis will exceed a bit the maximum value (1D)
            "keep" - let matlotlib figure this out
            tuple - standard parameter for set_ylim

    See Also
    --------
    plt.Axes.set_xlim, plt.Axes.set_ylim, plt.Axes.set_xscale, plt.Axes.set_yscale
    """
    xscale = kwargs.pop("xscale", None)
    yscale = kwargs.pop("yscale", None)
    ylim = kwargs.pop("ylim", "auto")
    xlim = kwargs.pop("xlim", "auto")

    if ylim is not "keep":
        if isinstance(ylim, tuple):
            pass
        elif ylim:
            ylim = ax.get_ylim()
            if h1.ndim == 1:
                if data.size > 0 and data.max() > 0:
                    ylim = (0, max(ylim[1], data.max() + (data.max() - ylim[0]) * 0.1))
                if yscale == "log":
                    ylim = (abs(data[data > 0].min()) * 0.9, ylim[1] * 1.1)
            elif h1.ndim == 2:
                if h1.shape[1] >= 2:
                    ylim = (h1.get_bin_left_edges(1)[0], h1.get_bin_right_edges(1)[-1])
                    if yscale == "log":
                        if ylim[0] <= 0:
                            raise RuntimeError("Cannot use logarithmic scale for non-positive bins.")
            else:
                raise RuntimeError("Invalid dimension: {0}".format(h1.ndim))
        ax.set_ylim(ylim)

    if xlim is not "keep":
        if isinstance(xlim, tuple):
            pass
        elif xlim:
            xlim = ax.get_xlim()
            if h1.shape[0] >= 2:
                if h1.ndim == 1:
                    xlim = (h1.bin_left_edges[0], h1.bin_right_edges[-1])
                elif h1.ndim == 2:
                    xlim = (h1.get_bin_left_edges(0)[0], h1.get_bin_right_edges(0)[-1])
                else:
                    raise RuntimeError("Invalid dimension: {0}".format(h1.ndim))
                if xscale == "log":
                    if xlim[0] <= 0:
                        raise RuntimeError("Cannot use logarithmic scale for non-positive bins.")
        ax.set_xlim(xlim)

    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)


def _add_ticks(ax, h1, kwargs):
    """Customize ticks for an axis (1D histogram).

    Parameters
    ----------
    ax : plt.Axes
    h1 :
    ticks: {"center", "edge"}, optional
    """
    ticks = kwargs.pop("ticks", None)
    if not ticks:
        return
    elif ticks == "center":
        ax.set_xticks(h1.bin_centers)
    elif ticks == "edge":
        ax.set_xticks(h1.bin_left_edges)
