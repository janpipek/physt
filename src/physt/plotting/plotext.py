import typing

import plotext as plt

if typing.TYPE_CHECKING:
    from physt.types import Histogram1D


types: typing.Tuple[str, ...] = ("bar", "hbar")


dims = {
    "hbar": [1],
    "bar": [1],
}


def hbar(
    h1: "Histogram1D",
) -> None:
    fig = plt.active()
    x = h1.bin_centers
    y = h1.frequencies
    fig.monitor.draw_bar(x, y, orientation="horizontal")
    plt.show()


def bar(
    h1: "Histogram1D",
) -> None:
    fig = plt.active()
    x = h1.bin_centers
    y = h1.frequencies
    fig.monitor.draw_bar(x, y, orientation="vertical")
    plt.show()
