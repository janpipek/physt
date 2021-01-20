"""ASCII plots (experimental).

The plots are printed directly to standard output.

"""
import typing

if typing.TYPE_CHECKING:
    from physt.histogram_nd import Histogram2D

try:
    import asciiplotlib

    ENABLE_ASCIIPLOTLIB = True
except ImportError:
    asciiplotlib = None
    ENABLE_ASCIIPLOTLIB = False

types: typing.Tuple[str, ...] = ("hbar",)

dims = {
    "hbar": [1],
}


def hbar(h1, width=80, show_values=False):
    if ENABLE_ASCIIPLOTLIB:
        data = h1.frequencies
        edges = h1.numpy_bins
        fig = asciiplotlib.figure()
        fig.hist(data, edges, orientation="horizontal")
        fig.show()
    else:
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

    def map(h2: "Histogram2D", **kwargs):
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
            raise ValueError(
                "Unsupported colormap: {0}, select from: {1}".format(cmap, SUPPORTED_CMAPS)
            )
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
