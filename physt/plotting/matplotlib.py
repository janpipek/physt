"""Matplotlib backend for plotting in physt.

This module contains several plotting functions and a lot of underscored
helper functions. User is expected to use only the former ones.

Plot functions for 1D histograms
- bar
- scatter
- fill
- line
- step

Plot functions for 2D histograms
- map
- image
- bar3d
- polar_map
- surface_map
- globe_map (for DirectionalHistogram)
- cylinder_map (for CylinderSurfaceHistogram)

Each plotting method supports many parameters. These are quite common
and very often corresponding to a matplotlib parameter of a same name.
The very general keyword argument dict is sequentially forwarded to
plotting helper functions that do part of the plotting job and popped of the used
parameters.

Parameters
----------

"""
from functools import wraps
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from physt.histogram1d import Histogram1D
from physt.histogram_nd import Histogram2D
from physt.histogram_collection import HistogramCollection
from physt.config import config
from physt.plotting.common import get_data, get_err_data, pop_kwargs_with_prefix, check_ndim
from physt.special_histograms import (
    CylindricalSurfaceHistogram,
    SphericalSurfaceHistogram,
)

# To be filled by register function
types = []
dims = {}


default_dpi = 72
default_figsize = matplotlib.rcParams["figure.figsize"]
default_cmap = "Greys"  # matplotlib.rcParams['image.cmap']


def register(*dim: int, use_3d: bool = False, use_polar: bool = False, collection: bool = False):
    """Decorator to wrap common plotting functionality.

    Parameters
    ----------
    dim : Dimensionality of histogram for which it is applicable
    use_3d : If True, the figure will be 3D.
    use_polar : If True, the figure will be in polar coordinates.
    collection : Whether to allow histogram collections to be used
    """
    if use_3d and use_polar:
        raise RuntimeError("Cannot have polar and 3d coordinates simultaneously.")

    # TODO: Add some kind of class parameter

    def wrapper(f):
        types.append(f.__name__)
        dims[f.__name__] = dim

        @wraps(f)
        @check_ndim(dim)
        def wrapped(hist, write_to: Optional[str] = None, dpi: Optional[float] = None, **kwargs):
            fig, ax = _get_axes(kwargs, use_3d=use_3d, use_polar=use_polar)

            if collection and isinstance(hist, HistogramCollection):
                title = kwargs.pop("title", hist.title)
                if not hist:
                    raise ValueError("Cannot plot empty histogram collection")
                for h in hist:
                    # TODO: Add some mechanism for argument maps (like sklearn?)
                    f(h, ax=ax, **kwargs)
                ax.legend()
                ax.set_title(title)
            else:
                f(hist, ax=ax, **kwargs)

            if write_to:
                fig = ax.figure
                fig.tight_layout()
                fig.savefig(write_to, dpi=dpi or default_dpi)
            return ax

        return wrapped

    return wrapper


@register(1, collection=True)
def bar(
    h1: Histogram1D, ax: Axes, *, errors: bool = False, **kwargs
):  # pylint: disable=blacklisted-name
    """Bar plot of 1D histograms."""
    show_stats = kwargs.pop("show_stats", False)
    show_values = kwargs.pop("show_values", False)
    value_format = kwargs.pop("value_format", None)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)
    label = kwargs.pop("label", h1.name)
    lw = kwargs.pop("linewidth", kwargs.pop("lw", 0.5))
    text_kwargs = pop_kwargs_with_prefix("text_", kwargs)

    data = get_data(h1, cumulative=cumulative, density=density)

    if "cmap" in kwargs:
        cmap = _get_cmap(kwargs)
        _, cmap_data = _get_cmap_data(data, kwargs)
        colors = cmap(cmap_data)
    else:
        colors = kwargs.pop("color", kwargs.pop("c", None))

    _apply_xy_lims(ax, h1, data, kwargs)
    _add_ticks(ax, h1, kwargs)

    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        kwargs["yerr"] = err_data
        if "ecolor" not in kwargs:
            kwargs["ecolor"] = "black"

    _add_labels(ax, h1, kwargs)
    ax.bar(
        h1.bin_left_edges,
        data,
        h1.bin_widths,
        align="edge",
        label=label,
        color=colors,
        linewidth=lw,
        **kwargs,
    )

    if show_values:
        _add_values(ax, h1, data, value_format=value_format, **text_kwargs)
    if show_stats:
        _add_stats_box(h1, ax, stats=show_stats)


@register(1, collection=True)
def scatter(
    h1: Histogram1D,
    ax: Axes,
    *,
    errors: bool = False,
    show_stats: bool = False,
    show_values: bool = False,
    density: bool = False,
    cumulative: bool = False,
    **kwargs,
):
    """Scatter plot of 1D histogram."""
    value_format = kwargs.pop("value_format", None)
    text_kwargs = pop_kwargs_with_prefix("text_", kwargs)
    label = kwargs.pop("label", h1.name)

    data = get_data(h1, cumulative=cumulative, density=density)

    if "cmap" in kwargs:
        cmap = _get_cmap(kwargs)
        _, cmap_data = _get_cmap_data(data, kwargs)
        kwargs["color"] = cmap(cmap_data)
    elif "color" in kwargs or "c" in kwargs:
        kwargs["color"] = kwargs.pop("color", kwargs.get("c", None))

    _apply_xy_lims(ax, h1, data, kwargs)
    _add_ticks(ax, h1, kwargs)
    _add_labels(ax, h1, kwargs)

    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        ax.errorbar(
            h1.bin_centers,
            data,
            yerr=err_data,
            fmt=kwargs.pop("fmt", "o"),
            ecolor=kwargs.pop("ecolor", "black"),
            ms=0,
        )
    ax.scatter(h1.bin_centers, data, label=label, **kwargs)

    if show_values:
        _add_values(ax, h1, data, value_format=value_format, **text_kwargs)
    if show_stats:
        _add_stats_box(h1, ax, stats=show_stats)


@register(1, collection=True)
def line(
    h1: Histogram1D,
    ax: Axes,
    *,
    show_stats: bool = False,
    show_values: bool = False,
    density: bool = False,
    cumulative: bool = False,
    errors: bool = False,
    **kwargs,
):
    """Line plot of 1D histogram."""
    value_format = kwargs.pop("value_format", None)
    text_kwargs = pop_kwargs_with_prefix("text_", kwargs)
    kwargs["label"] = kwargs.get("label", h1.name)

    data = get_data(h1, cumulative=cumulative, density=density)
    _apply_xy_lims(ax, h1, data, kwargs)
    _add_ticks(ax, h1, kwargs)
    _add_labels(ax, h1, kwargs)

    if errors:
        err_data = get_err_data(h1, cumulative=cumulative, density=density)
        ax.errorbar(
            h1.bin_centers,
            data,
            yerr=err_data,
            fmt=kwargs.pop("fmt", "-"),
            ecolor=kwargs.pop("ecolor", "black"),
            **kwargs,
        )
    else:
        ax.plot(h1.bin_centers, data, **kwargs)

    if show_stats:
        _add_stats_box(h1, ax, stats=show_stats)
    if show_values:
        _add_values(ax, h1, data, value_format=value_format, **text_kwargs)


@register(1, collection=True)
def fill(h1: Histogram1D, ax: Axes, **kwargs):
    """Fill plot of 1D histogram."""
    show_stats = kwargs.pop("show_stats", False)
    # show_values = kwargs.pop("show_values", False)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)
    kwargs["label"] = kwargs.get("label", h1.name)

    data = get_data(h1, cumulative=cumulative, density=density)
    _apply_xy_lims(ax, h1, data, kwargs)
    _add_ticks(ax, h1, kwargs)
    _add_labels(ax, h1, kwargs)

    ax.fill_between(h1.bin_centers, 0, data, **kwargs)

    if show_stats:
        _add_stats_box(h1, ax, stats=show_stats)
    # if show_values:
    #     _add_values(ax, h1, data)
    return ax


@register(1, collection=True)
def step(h1: Histogram1D, ax: Axes, **kwargs):
    """Step line-plot of 1D histogram."""
    show_stats = kwargs.pop("show_stats", False)
    show_values = kwargs.pop("show_values", False)
    density = kwargs.pop("density", False)
    cumulative = kwargs.pop("cumulative", False)
    value_format = kwargs.pop("value_format", None)
    text_kwargs = pop_kwargs_with_prefix("text_", kwargs)
    kwargs["label"] = kwargs.get("label", h1.name)

    data = get_data(h1, cumulative=cumulative, density=density)
    _apply_xy_lims(ax, h1, data, kwargs)
    _add_ticks(ax, h1, kwargs)
    _add_labels(ax, h1, kwargs)

    ax.step(h1.numpy_bins, np.concatenate([data[:1], data]), **kwargs)

    if show_stats:
        _add_stats_box(h1, ax, stats=show_stats)
    if show_values:
        _add_values(ax, h1, data, value_format=value_format, **text_kwargs)


@register(2)
def map(
    h2: Histogram2D,
    ax: Axes,
    *,
    show_zero: bool = True,
    show_values: bool = False,
    show_colorbar: bool = True,
    value_format: Any = str,
    x=None,
    y=None,
    **kwargs,
):
    """Coloured-rectangle plot of 2D histogram.

    Parameters
    ----------
    show_zero : Whether to show coloured box for bins with 0 frequency (otherwise background).
    show_values : Whether to show labels with frequencies/densities in the middle of the bin


    text_color : Optional
        Colour of text descriptions
    text_alpha : Optional[float]
        Alpha for the text labels only
    x : Optional[Callable]
        Transformation of x bin coordinates
    y : Optional[Callable]
        Transformation of y bin coordinates
    zorder : float
        z-order in the axis (higher number above lower)

    See Also
    --------
    image, polar_map, surface_map

    Notes
    -----
    If you transform axes using x or y parameters, the deduction of axis limits
    does not work well automatically. Please, make sure to attend to it yourself.
    The densities in transformed maps are calculated from original bins.
    """
    # Detect transformation
    transformed = False
    if x is not None or y is not None:
        if not x:
            x = lambda x, y: x
        if not y:
            y = lambda x, y: y
        transformed = True

    # TODO: Implement correctly the text_kwargs

    if isinstance(value_format, str):
        format_str = "{0:" + value_format + "}"
        value_format = format_str.format

    rect_args = {}
    if "zorder" in kwargs:
        rect_args["zorder"] = kwargs.pop("zorder")

    data = get_data(h2, cumulative=False, flatten=True, density=kwargs.pop("density", False))

    cmap = _get_cmap(kwargs)
    norm, cmap_data = _get_cmap_data(data, kwargs)
    colors = cmap(cmap_data)

    xpos, ypos = (arr.flatten() for arr in h2.get_bin_left_edges())
    dx, dy = (arr.flatten() for arr in h2.get_bin_widths())
    text_x, text_y = (arr.flatten() for arr in h2.get_bin_centers())

    _apply_xy_lims(ax, h2, data=data, kwargs=kwargs)
    _add_labels(ax, h2, kwargs)

    ax.autoscale_view()

    alphas = _get_alpha_data(cmap_data, kwargs)
    if np.isscalar(alphas):
        alphas = np.ones_like(data) * alphas

    for i in range(len(xpos)):
        bin_color = colors[i]
        alpha = alphas[i]

        if data[i] != 0 or show_zero:
            if not transformed:
                rect = plt.Rectangle(
                    (xpos[i], ypos[i]),
                    dx[i],
                    dy[i],
                    facecolor=bin_color,
                    edgecolor=kwargs.get("grid_color", cmap(0.5)),
                    lw=kwargs.get("lw", 0.5),
                    alpha=alpha,
                    **rect_args,
                )
                tx, ty = text_x[i], text_y[i]

            else:
                # See http://matplotlib.org/users/path_tutorial.html
                points = (
                    (xpos[i], ypos[i]),
                    (xpos[i] + dx[i], ypos[i]),
                    (xpos[i] + dx[i], ypos[i] + dy[i]),
                    (xpos[i], ypos[i] + dy[i]),
                    (xpos[i], ypos[i]),
                )

                verts = [(x(*p), y(*p)) for p in points]

                codes = [
                    path.Path.MOVETO,
                    path.Path.LINETO,
                    path.Path.LINETO,
                    path.Path.LINETO,
                    path.Path.CLOSEPOLY,
                ]

                rect_path = path.Path(verts, codes)
                rect = patches.PathPatch(
                    rect_path,
                    facecolor=bin_color,
                    edgecolor=kwargs.get("grid_color", cmap(0.5)),
                    lw=kwargs.get("lw", 0.5),
                    alpha=alpha,
                    **rect_args,
                )

                tx = x(text_x[i], text_y[i])
                ty = y(text_x[i], text_y[i])
            ax.add_patch(rect)

            if show_values:
                text = value_format(data[i])
                yiq_y = np.dot(bin_color[:3], [0.299, 0.587, 0.114])

                text_color = kwargs.get("text_color", None)
                if not text_color:
                    if yiq_y > 0.5:
                        text_color = (0.0, 0.0, 0.0, kwargs.get("text_alpha", alpha))
                    else:
                        text_color = (1.0, 1.0, 1.0, kwargs.get("text_alpha", alpha))
                ax.text(
                    tx,
                    ty,
                    text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=text_color,
                    clip_on=True,
                    **rect_args,
                )

    if show_colorbar:
        _add_colorbar(ax, cmap, cmap_data, norm)


@register(2, use_3d=True)
def bar3d(h2: Histogram2D, ax: Axes3D, *, density: bool = False, **kwargs):
    """Plot of 2D histograms as 3D boxes."""
    data = get_data(h2, cumulative=False, flatten=True, density=density)

    if "cmap" in kwargs:
        cmap = _get_cmap(kwargs)
        _, cmap_data = _get_cmap_data(data, kwargs)
        colors = cmap(cmap_data)
    else:
        colors = kwargs.pop("color", kwargs.pop("c", "blue"))

    xpos, ypos = (arr.flatten() for arr in h2.get_bin_centers())
    zpos = np.zeros_like(ypos)
    dx, dy = (arr.flatten() for arr in h2.get_bin_widths())

    _add_labels(ax, h2, kwargs)
    ax.bar3d(xpos, ypos, zpos, dx, dy, data, color=colors, **kwargs)
    ax.set_zlabel("density" if density else "frequency")


@register(2)
def image(
    h2: Histogram2D,
    ax: Axes,
    *,
    show_colorbar: bool = True,
    interpolation: str = "nearest",
    **kwargs,
):
    """Plot of 2D histograms based on pixmaps.

    Similar to map, but it:
    - has fewer options
    - is much more effective (enables thousands)
    - does not support irregular bins

    Parameters
    ----------
    interpolation: interpolation parameter passed to imshow, default: "nearest" (creates rectangles)
    """
    cmap = _get_cmap(kwargs)  # h2 as well?
    data = get_data(h2, cumulative=False, density=kwargs.pop("density", False))
    norm, cmap_data = _get_cmap_data(data, kwargs)
    # zorder = kwargs.pop("zorder", None)

    for binning in h2._binnings:
        if not binning.is_regular():
            raise RuntimeError(
                "Histograms with irregular bins cannot be plotted using image method."
            )

    kwargs["interpolation"] = interpolation
    if kwargs.get("xscale") == "log" or kwargs.get("yscale") == "log":
        raise RuntimeError("Cannot use logarithmic axes with image plots.")

    _apply_xy_lims(ax, h2, data=data, kwargs=kwargs)

    _add_labels(ax, h2, kwargs)
    ax.imshow(
        data.T[::-1, :],
        cmap=cmap,
        norm=norm,
        extent=(
            h2.bins[0][0, 0],
            h2.bins[0][-1, 1],
            h2.bins[1][0, 0],
            h2.bins[1][-1, 1],
        ),
        aspect="auto",
        **kwargs,
    )

    if show_colorbar:
        _add_colorbar(ax, cmap, cmap_data, norm)


@register(2, use_polar=True)
def polar_map(
    hist: Histogram2D, ax: Axes, *, show_zero: bool = True, show_colorbar: bool = True, **kwargs
):
    """Polar map of polar histograms.

    Similar to map, but supports less parameters."""
    data = get_data(hist, cumulative=False, flatten=True, density=kwargs.pop("density", False))

    cmap = _get_cmap(kwargs)
    norm, cmap_data = _get_cmap_data(data, kwargs)
    colors = cmap(cmap_data)

    r, phi = (arr.flatten() for arr in hist.get_bin_left_edges())
    delta_r, delta_phi = (arr.flatten() for arr in hist.get_bin_widths())
    r_max, _ = (arr.flatten() for arr in hist.get_bin_right_edges())

    bar_args = {}
    if "zorder" in kwargs:
        bar_args["zorder"] = kwargs.pop("zorder")

    alphas = _get_alpha_data(cmap_data, kwargs)
    if np.isscalar(alphas):
        alphas = np.ones_like(data) * alphas

    for i in range(len(r)):
        if data[i] > 0 or show_zero:
            bin_color = colors[i]
            ax.bar(
                phi[i],
                delta_r[i],
                width=delta_phi[i],
                bottom=r[i],
                align="edge",
                color=bin_color,
                edgecolor=kwargs.get("grid_color", cmap(0.5)),
                lw=kwargs.get("lw", 0.5),
                alpha=alphas[i],
                **bar_args,
            )

    ax.set_rmax(r_max.max())
    if show_colorbar:
        _add_colorbar(ax, cmap, cmap_data, norm)


@register(2, use_3d=True)
def globe_map(
    hist: Union[Histogram2D, SphericalSurfaceHistogram],
    ax: Axes3D,
    *,
    lw: int = 1,
    show_zero: bool = True,
    **kwargs,
):
    """Heat map plotted on the surface of a sphere."""
    data = get_data(hist, cumulative=False, flatten=False, density=kwargs.pop("density", False))

    cmap = _get_cmap(kwargs)
    norm, cmap_data = _get_cmap_data(data, kwargs)
    colors = cmap(cmap_data)

    r = 1
    xs = r * np.outer(np.sin(hist.numpy_bins[0]), np.cos(hist.numpy_bins[1]))
    ys = r * np.outer(np.sin(hist.numpy_bins[0]), np.sin(hist.numpy_bins[1]))
    zs = r * np.outer(np.cos(hist.numpy_bins[0]), np.ones(hist.shape[1] + 1))

    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if not show_zero and not data[i, j]:
                continue
            x = xs[i, j], xs[i, j + 1], xs[i + 1, j + 1], xs[i + 1, j]
            y = ys[i, j], ys[i, j + 1], ys[i + 1, j + 1], ys[i + 1, j]
            z = zs[i, j], zs[i, j + 1], zs[i + 1, j + 1], zs[i + 1, j]
            verts = [list(zip(x, y, z))]
            col = Poly3DCollection(verts)
            col.set_facecolor(colors[i, j])
            col.set_edgecolor("black")
            col.set_linewidth(lw)
            ax.add_collection3d(col)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if matplotlib.__version__ < "2":
        ax.plot_surface([], [], [], color="b")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    return ax


@register(2, use_3d=True)
def cylinder_map(
    hist: Union[Histogram2D, CylindricalSurfaceHistogram],
    ax: Axes3D,
    *,
    show_zero: bool = True,
    **kwargs,
):
    """Heat map plotted on the surface of a cylinder."""
    data = get_data(hist, cumulative=False, flatten=False, density=kwargs.pop("density", False))

    cmap = _get_cmap(kwargs)
    norm, cmap_data = _get_cmap_data(data, kwargs)
    colors = cmap(cmap_data)

    r = kwargs.pop("radius", getattr(hist, "radius", 1))

    xs = r * np.outer(np.cos(hist.numpy_bins[0]), np.ones(hist.shape[1] + 1))
    ys = r * np.outer(np.sin(hist.numpy_bins[0]), np.ones(hist.shape[1] + 1))
    zs = np.outer(np.ones(hist.shape[0] + 1), hist.numpy_bins[1])

    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if not show_zero and not data[i, j]:
                continue
            x = xs[i, j], xs[i, j + 1], xs[i + 1, j + 1], xs[i + 1, j]
            y = ys[i, j], ys[i, j + 1], ys[i + 1, j + 1], ys[i + 1, j]
            z = zs[i, j], zs[i, j + 1], zs[i + 1, j + 1], zs[i + 1, j]
            verts = [list(zip(x, y, z))]
            col = Poly3DCollection(verts)
            col.set_facecolor(colors[i, j])
            ax.add_collection3d(col)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if matplotlib.__version__ < "2":
        ax.plot_surface([], [], [], color="b")
    ax.set_xlim(-r * 1.1, r * 1.1)
    ax.set_ylim(-r * 1.1, r * 1.1)
    ax.set_zlim(zs.min(), zs.max())


@register(2, use_3d=True)
def surface_map(
    hist,
    ax: Axes3D,
    *,
    show_zero: bool = True,
    x=(lambda x, y: x),
    y=(lambda x, y: y),
    z=(lambda x, y: 0),
    **kwargs,
):
    """Coloured-rectangle plot of 2D histogram, placed on an arbitrary surface.

    Each bin is mapped to a rectangle in 3D space using the x,y,z functions.

    Parameters
    ----------
    hist : Histogram2D
    show_zero : Optional[bool]
        Whether to show coloured box for bins with 0 frequency (otherwise background).
    x : function
        Function with 2 parameters used to map bins to spatial x coordinate
    y : function
        Function with 2 parameters used to map bins to spatial y coordinate
    z : function
        Function with 2 parameters used to map bins to spatial z coordinate

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot

    See Also
    --------
    map, cylinder_map, globe_map
    """
    data = get_data(hist, cumulative=False, flatten=False, density=kwargs.pop("density", False))

    cmap = _get_cmap(kwargs)
    norm, cmap_data = _get_cmap_data(data, kwargs)
    colors = cmap(cmap_data)

    xs = np.ndarray((hist.shape[0] + 1, hist.shape[1] + 1), dtype=float)
    ys = np.ndarray((hist.shape[0] + 1, hist.shape[1] + 1), dtype=float)
    zs = np.ndarray((hist.shape[0] + 1, hist.shape[1] + 1), dtype=float)

    edges_x = hist.numpy_bins[0]
    edges_y = hist.numpy_bins[1]

    for i in range(hist.shape[0] + 1):
        for j in range(hist.shape[1] + 1):
            xs[i, j] = x(edges_x[i], edges_y[j])
            ys[i, j] = y(edges_x[i], edges_y[j])
            zs[i, j] = z(edges_x[i], edges_y[j])

    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if not show_zero and not data[i, j]:
                continue
            x = xs[i, j], xs[i, j + 1], xs[i + 1, j + 1], xs[i + 1, j]
            y = ys[i, j], ys[i, j + 1], ys[i + 1, j + 1], ys[i + 1, j]
            z = zs[i, j], zs[i, j + 1], zs[i + 1, j + 1], zs[i + 1, j]
            verts = [list(zip(x, y, z))]
            col = Poly3DCollection(verts)
            col.set_facecolor(colors[i, j])
            ax.add_collection3d(col)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if matplotlib.__version__ < "2":
        ax.plot_surface([], [], [], color="b")  # Dummy plot
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(zs.min(), zs.max())

    # ax.plot_surface(x, y, z, rstride=hist.shape[0], color="b")

    return ax


def pair_bars(
    first: Histogram1D,
    second: Histogram1D,
    *,
    orientation: str = "vertical",
    kind: str = "bar",
    **kwargs,
) -> Axes:
    """Draw two different histograms mirrored in one figure.

    Parameters
    ----------
    first:
    second:
    color1: Color for the first histogram
    color2: Color for the second histogram
    orientation: vertical (not enabled yet) or horizontal
    """
    # TODO: enable vertical as well as horizontal
    _, ax = _get_axes(kwargs)
    color1 = kwargs.pop("color1", "red")
    color2 = kwargs.pop("color2", "blue")
    title = kwargs.pop("title", "{0} - {1}".format(first.name, second.name))
    xlim = kwargs.pop(
        "xlim",
        (
            min(first.bin_left_edges[0], first.bin_left_edges[0]),
            max(first.bin_right_edges[-1], second.bin_right_edges[-1]),
        ),
    )

    with config.enable_free_arithmetics():
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


def _get_axes(
    kwargs: Dict[str, Any], *, use_3d: bool = False, use_polar: bool = False
) -> Tuple[Figure, Union[Axes, Axes3D]]:
    """Prepare the axis to draw into.

    Parameters
    ----------
    use_3d: If True, an axis with 3D projection is created.
    use_polar: If True, the plot will have polar coordinates.

    Kwargs
    ------
    ax: Optional[plt.Axes]
        An already existing axis to be used.
    figsize: Optional[tuple]
        Size of the new figure (if no axis is given).

    Returns
    ------
    fig : plt.Figure
    ax : plt.Axes | Axes3D
    """
    figsize = kwargs.pop("figsize", default_figsize)
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
        fig = ax.get_figure()
    elif use_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    elif use_polar:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _get_cmap(kwargs: dict) -> colors.Colormap:
    """Get the colour map for plots that support it.

    Parameters
    ----------
    cmap : str or colors.Colormap or list of colors
        A map or an instance of cmap. This can also be a seaborn palette
        (if seaborn is installed).
    """
    from matplotlib.colors import ListedColormap

    cmap = kwargs.pop("cmap", default_cmap)
    if isinstance(cmap, list):
        return ListedColormap(cmap)
    if isinstance(cmap, str):
        try:
            return plt.get_cmap(cmap)
        except BaseException:
            try:
                # Try to use seaborn palette
                import seaborn as sns

                sns_palette = sns.color_palette(cmap, n_colors=256)
                return ListedColormap(sns_palette, name=cmap)
            except ImportError:
                pass
        raise ValueError(f"Cmap '{cmap}' could not be found.")
    return cmap


def _get_cmap_data(data, kwargs) -> Tuple[colors.Normalize, np.ndarray]:
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


def _get_alpha_data(data: np.ndarray, kwargs) -> np.ndarray:
    """Get alpha values for all data points.

    Parameters
    ----------
    alpha: Callable or float
        This can be a fixed value or a function of the data.
    """
    alpha = kwargs.pop("alpha", 1)
    if hasattr(alpha, "__call__"):
        return np.vectorize(alpha)(data)
    if np.isscalar(alpha):
        return np.ones_like(data) * alpha
    return alpha


def _add_labels(ax: Axes, h: Union[Histogram1D, Histogram2D], kwargs: dict):
    """Add axis and plot labels.

    TODO: Document kwargs
    """
    title = kwargs.pop("title", h.title)
    xlabel = kwargs.pop("xlabel", h.axis_names[0])
    ylabel = kwargs.pop("ylabel", h.axis_names[1] if len(h.axis_names) == 2 else None)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.get_figure().tight_layout()


def _add_values(ax: Axes, h1: Histogram1D, data, *, value_format=str, **kwargs):
    """Show values next to each bin in a 1D plot.

    Parameters
    ----------
    ax : plt.Axes
    h1 : physt.histogram1d.Histogram1D
    data : array_like
        The values to be displayed
    value_format : foratting string or callable to create the value representation
    kwargs : dict
        Parameters to be passed to matplotlib to override standard text params.
    """
    from .common import get_value_format

    value_format = get_value_format(value_format)
    text_kwargs = {"ha": "center", "va": "bottom", "clip_on": True}
    text_kwargs.update(kwargs)

    for x, y in zip(h1.bin_centers, data):
        ax.text(x, y, str(value_format(y)), **text_kwargs)


def _add_colorbar(ax: Axes, cmap: colors.Colormap, cmap_data: np.ndarray, norm: colors.Normalize):
    """Show a colorbar right of the plot."""
    fig = ax.get_figure()
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(cmap_data)  # TODO: Or what???
    fig.colorbar(mappable, ax=ax)


def _add_stats_box(h1: Histogram1D, ax: Axes, stats: Union[str, bool] = "all"):
    """Insert a small legend-like box with statistical information.

    Parameters
    ----------
    stats : "all" | "total" | True
        What info to display

    Note
    ----
    Very basic implementation.
    """

    # place a text box in upper left in axes coords
    if stats in ["all", True]:
        text_frags = [f"Total: {h1.total}"]

        mean = h1.mean()
        if mean is not None:
            text_frags.append(f"Mean: {mean:.2f}")
        std = h1.std()
        if std is not None:
            text_frags.append(f"Std.dev: {std:.2f}")
        text = "\n".join(text_frags)
    elif stats == "total":
        text = f"Total: {h1.total}"
    else:
        raise ValueError("Invalid stats specification")

    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )


def _apply_xy_lims(ax: Axes, h: Union[Histogram1D, Histogram2D], data: np.ndarray, kwargs: dict):
    """Apply axis limits and scales from kwargs.

    Note: if exponential binning is used, the scale defaults to "log"

    Parameters
    ----------
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
        ylim : { "keep", "auto" } or tuple(float)
            "auto" (default)
                - the axis will fit first and last bin edges (2D)
                - the axis will exceed a bit the maximum value (1D)
            "keep" - let matlotlib figure this out
            tuple - standard parameter for set_ylim
        invert_y : Optional[bool]
            If True, higher values go down

    See Also
    --------
    plt.Axes.set_xlim, plt.Axes.set_ylim, plt.Axes.set_xscale, plt.Axes.set_yscale
    """
    ylim = kwargs.pop("ylim", "auto")
    xlim = kwargs.pop("xlim", "auto")
    invert_y = kwargs.pop("invert_y", False)

    from physt.binnings import ExponentialBinning

    # First, get the axis scaling
    if h.ndim == 1:
        xscale = kwargs.pop(
            "xscale", "log" if isinstance(h.binnings[0], ExponentialBinning) else None
        )
        yscale = kwargs.pop("yscale", None)
    elif h.ndim == 2:
        xscale = kwargs.pop(
            "xscale", "log" if isinstance(h.binnings[0], ExponentialBinning) else None
        )
        yscale = kwargs.pop(
            "yscale", "log" if isinstance(h.binnings[1], ExponentialBinning) else None
        )
    else:
        raise ValueError(f"Invalid histogram dimension: {h.ndim}")

    if ylim != "keep":
        if isinstance(ylim, tuple):
            pass
        elif ylim:
            ylim = ax.get_ylim()
            if h.ndim == 1:
                if data.size > 0 and data.max() > 0:
                    ylim = (0, max(ylim[1], data.max() + (data.max() - ylim[0]) * 0.1))
                if yscale == "log":
                    ylim = (abs(data[data > 0].min()) * 0.9, ylim[1] * 1.1)
            elif h.ndim == 2:
                if h.shape[1] >= 2:
                    ylim = (h.get_bin_left_edges(1)[0], h.get_bin_right_edges(1)[-1])
                    if yscale == "log":
                        if ylim[0] <= 0:
                            raise ValueError("Cannot use logarithmic scale for non-positive bins.")
            if invert_y:
                ylim = ylim[::-1]
                # ax.xaxis.tick_top()
                # ax.xaxis.set_label_position('top')
        ax.set_ylim(ylim)

    if xlim != "keep":
        if isinstance(xlim, tuple):
            pass
        elif xlim:
            xlim = ax.get_xlim()
            if h.shape[0] >= 1:
                if h.ndim <= 2:
                    xlim = (h.get_bin_left_edges(0)[0], h.get_bin_right_edges(0)[-1])
                else:
                    raise ValueError("Invalid dimension: {0}".format(h.ndim))
                if xscale == "log":
                    if xlim[0] <= 0:
                        raise ValueError("Cannot use xscale='log' for non-positive bins.")
        ax.set_xlim(*xlim)

    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)


def _add_ticks(ax: Axes, h1: Histogram1D, kwargs: dict):
    """Customize ticks for an axis (1D histogram).

    Parameters
    ----------
    ticks: {"center", "edge"}, optional
        Position of the ticks
    tick_handler: Callable[[Histogram1D, float, float], Tuple[List[float], List[str]]]
        ...
    """
    ticks = kwargs.pop("ticks", None)
    tick_handler = kwargs.pop("tick_handler", None)

    if tick_handler:
        if ticks:
            raise ValueError("Cannot specify both tick and tick_handler")
        ticks, labels = tick_handler(h1, *ax.get_xlim())
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    if ticks == "center":
        ax.set_xticks(h1.bin_centers)
    if ticks == "edge":
        ax.set_xticks(h1.bin_left_edges)
