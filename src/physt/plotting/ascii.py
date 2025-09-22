"""ASCII plots (experimental).

The plots are printed directly to standard output.

"""

from __future__ import annotations

import typing

import numpy as np
import rich.color
import rich.console
from rich.style import Style
from rich.text import Text

from physt.plotting.common import get_value_format

if typing.TYPE_CHECKING:
    from typing import Any, Optional, Union

    from physt.types import Histogram1D, Histogram2D


types: typing.Tuple[str, ...] = ("hbar", "map")

dims = {
    "hbar": [1],
    "map": [2],
}


def hbar(
    h1: "Histogram1D",
    *,
    max_width: Optional[int] = None,
    show_values: bool = False,
    show_labels: bool = False,
    color: Optional[str] = None,
    label_width: int = 10,
) -> None:
    """Horizontal bar plot in block characters.

    :param h1: Histogram to plot.
    :param max_width: Width of the bars (including labels and values).
        By default, the width of the terminal minus one.
    :param show_values: Whether to show values right of the bars.
    :param show_labels: Whether to show bin labels left of the bars.
    :param label_width: Width of the label field (if shown).
    :param color: Color of the bars.
    """
    console = rich.console.Console()

    # Calculate width available for box characters
    max_width = max_width or (console.width - 1)
    if show_labels:
        max_width -= label_width + 1
    if show_values:
        max_width -= len(str(h1.frequencies.max())) + 1

    bar_widths = (h1.frequencies / h1.frequencies.max() * max_width).round().astype(int)
    style_kwargs: dict[str, Any] = {}
    if color:
        style_kwargs["color"] = color
    style = Style(**style_kwargs)

    for i in range(h1.bin_count):
        if show_labels:
            label = str(h1.bin_left_edges[i])
            console.print(
                Text(label.rjust(label_width)[:label_width], style=style), end=" "
            )
        if bar_widths[i] == 0:
            bar_text = LEFT_LINE_CHAR
        else:
            bar_text = FULL_SQUARE_CHAR * bar_widths[i]
        if show_values:
            console.print(Text(bar_text, style=style), h1.frequencies[i])
        else:
            console.print(Text(bar_text), style=style)
    if show_labels:
        last_edge = str(h1.max_edge)
        console.print(
            Text(last_edge.rjust(label_width)[:label_width], style=style), end=" "
        )
    console.print()


# TODO: Support more maps
SUPPORTED_CMAPS = ("Greys", "Greys_r")
DEFAULT_CMAP = SUPPORTED_CMAPS[1]

SHADING_CHARS = " ░▒▓█"
"""Characters used for shading in the ASCII map."""

FULL_SQUARE_CHAR = SHADING_CHARS[-1]
"""Character used for horizontal bar plots."""

LEFT_LINE_CHAR = "▏"
# TODO: Include more varying width characters for the bar


def map(
    h2: "Histogram2D", *, use_color: typing.Optional[bool] = None, **kwargs
) -> None:
    """Heat map.

    Depending on the color system, it uses either block characters or shades
    from the colormap.
    """

    console = rich.console.Console()
    color_system = console.color_system
    if use_color is None:
        use_color = bool(color_system)

    # Value format
    value_format = get_value_format(kwargs.pop("value_format", ".2f"))
    cmap_data = _get_cmap_data(h2.frequencies, kwargs)

    if use_color:

        def _render_cell(value: float) -> Text:
            color = rich.color.Color.from_rgb(
                int(255 * value), int(255 * value), int(255 * value)
            )
            return Text(FULL_SQUARE_CHAR, style=Style(color=color))

        # Colour map
        cmap = kwargs.pop("cmap", DEFAULT_CMAP)
        if cmap == "Greys":
            cmap_data = 1.0 - cmap_data
            colorbar_range = np.arange(h2.shape[1] + 1, -1, -1) / h2.shape[1]
        elif cmap == "Greys_r":
            colorbar_range = np.arange(h2.shape[1] + 1) / h2.shape[1]
        else:
            raise ValueError(
                f"Unsupported colormap: {cmap}, select from: {SUPPORTED_CMAPS}"
            )

    else:

        def _render_cell(value: float) -> Text:
            return Text(
                SHADING_CHARS[
                    int(
                        np.clip(value * (len(SHADING_CHARS)), 0, len(SHADING_CHARS) - 1)
                    )
                ]
            )

        colorbar_range = np.arange(h2.shape[1] + 1) / h2.shape[1]

    console.print(
        (value_format(h2.get_bin_right_edges(0)[-1].item()) + " →").rjust(
            h2.shape[0] + 2, " "
        )
    )
    console.print("┌" + "─" * h2.shape[0] + "┐")
    for y in range(h2.shape[1] - 1, -1, -1):
        line_frags: list[Union[Text, str]] = ["│"]
        line_frags += [_render_cell(cmap_data[x, y].item()) for x in range(h2.shape[0])]
        line_frags.append("│")
        if y == h2.shape[1] - 1:
            line_frags.append(value_format(h2.get_bin_right_edges(1)[-1].item()) + " ↑")
        if y == 0:
            line_frags.append(value_format(h2.get_bin_left_edges(1)[0].item()) + " ↓")
        console.print(*line_frags, sep="")
    console.print("└" + "─" * h2.shape[0] + "┘")
    console.print("←", value_format(h2.get_bin_left_edges(0)[0].item()))
    colorbar_frags = [_render_cell(j) for j in colorbar_range]
    console.print("↓", 0, sep="")
    console.print(*colorbar_frags, sep="")
    console.print(str(h2.frequencies.max()).rjust(h2.shape[1], " "), "↑")


def _get_cmap_data(data, kwargs) -> np.ndarray:
    """Get normalized values to be used with a colormap.

    Parameters
    ----------
    data : array_like
    cmap_min : Optional[float] or "min"
        By default 0. If "min", minimum value of the data.
    cmap_max : Optional[float]
        By default, maximum value of the data
    cmap_normalize : Optional[str]

    Returns
    -------
    normalized_data : array_like
    """
    norm = kwargs.pop("cmap_normalize", None)
    if norm == "log":
        cmap_max = np.log(kwargs.pop("cmap_max", data.max()))
        cmap_min = np.log(kwargs.pop("cmap_min", data[data > 0].min()))
        return (np.log(data) - cmap_min) / (cmap_max - cmap_min)
    elif not norm:
        cmap_max = kwargs.pop("cmap_max", data.max())
        cmap_min = kwargs.pop("cmap_min", 0)
        if cmap_min == "min":
            cmap_min = data.min()
        return (data - cmap_min) / (cmap_max - cmap_min)
    else:
        raise ValueError(f"Unsupported normalization: {norm}")
