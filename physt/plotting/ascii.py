"""ASCII plots (experimental).

The plots are printed directly to standard output.

"""
import typing

if typing.TYPE_CHECKING:
    from physt.types import Histogram1D, Histogram2D


types: typing.Tuple[str, ...] = ("hbar",)

dims = {
    "hbar": [1],
}


def hbar(h1: "Histogram1D", width: int = 80, show_values: bool = False) -> None:
    data = (h1.normalize().frequencies * width).round().astype(int)
    for i in range(h1.bin_count):
        if show_values:
            print("#" * data[i], h1.frequencies[i])
        else:
            print("#" * data[i])


try:
    import xtermcolor

    SUPPORTED_CMAPS = ("Greys", "Greys_r")
    DEFAULT_CMAP = SUPPORTED_CMAPS[1]

    def map(h2: "Histogram2D", **kwargs) -> None:
        """Heat map.

        Note: Available only if xtermcolor present.
        """

        # Value format
        val_format = kwargs.pop("value_format", ".2f")
        if isinstance(val_format, str):
            value_format = lambda val: (("{0:" + val_format + "}").format(val))

        data = (h2.frequencies / h2.frequencies.max() * 255).astype(int)

        # Colour map
        cmap = kwargs.pop("cmap", DEFAULT_CMAP)
        if cmap == "Greys":
            data = 255 - data
            colorbar_range = range(h2.shape[1] + 1, -1, -1)
        elif cmap == "Greys_r":
            colorbar_range = range(h2.shape[1] + 2)
        else:
            raise ValueError(f"Unsupported colormap: {cmap}, select from: {SUPPORTED_CMAPS}")
        colors = (65536 + 256 + 1) * data

        print((value_format(h2.get_bin_right_edges(0)[-1]) + " →").rjust(h2.shape[1] + 2, " "))
        print("+" + "-" * h2.shape[1] + "+")
        for i in range(h2.shape[0] - 1, -1, -1):
            line_frags = [
                xtermcolor.colorize("█", bg=0, rgb=colors[i, j]) for j in range(h2.shape[1])
            ]
            line = "|" + "".join(line_frags) + "|"
            if i == h2.shape[0] - 1:
                line += value_format(h2.get_bin_right_edges(1)[-1]) + " ↑"
            if i == 0:
                line += value_format(h2.get_bin_left_edges(1)[0]) + " ↓"
            print(line)
        print("+" + "-" * h2.shape[1] + "+")
        print("←", value_format(h2.get_bin_left_edges(0)[0]))
        colorbar_frags = [
            xtermcolor.colorize("█", bg=0, rgb=(65536 + 256 + 1) * int(j * 255 / (h2.shape[1] + 2)))
            for j in colorbar_range
        ]
        colorbar = "".join(colorbar_frags)
        print()
        print("↓", 0)
        print(colorbar)
        print(str(h2.frequencies.max()).rjust(h2.shape[1], " "), "↑")

    types = types + ("map",)
    dims["map"] = [2]
except ImportError:
    pass
