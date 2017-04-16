# -*- coding: utf-8 -*-
"""ASCII plots
"""
from __future__ import absolute_import

types = ("hbar",)

dims = {
    "hbar" : [1],
}

def hbar(h1, width=80, show_values=False):
    data = (h1.normalize().frequencies * width).round().astype(int)
    for i in range(h1.bin_count):
        if show_values:
            print("#" * data[i], h1.frequencies[i])
        else:
            print("#" * data[i])


try:
    import xtermcolor
    def map(h2, **kwargs):
        """Heat map

        Available only if xtermcolor present

        Parameters
        ----------
        h2 : physt.histogram_nd.Histogram2D
        """
        val_format = kwargs.pop("value_format", ".2f")
        if isinstance(val_format, str):
            value_format = lambda val: (("{0:" + val_format + "}").format(val))
        data = (h2.frequencies / h2.frequencies.max() * 255).astype(int)
        colors = (65536 + 256 + 1) * data
        print((value_format(h2.get_bin_right_edges(0)[-1]) + " →").rjust(h2.shape[1] + 2, " "))
        print("+" + "-" * h2.shape[1] + "+")
        for i in range(h2.shape[0]-1, -1, -1):
            line = [
                xtermcolor.colorize("█", bg=0, rgb=colors[i,j])
                # xtermcolor.colorize(" ", rgb="#000000", bg="#{0}{0}{0}".format(hex(i)[2:].zfill(2)))
                for j in range(h2.shape[1])
            ]
            line = "|" + "".join(line) + "|"
            if i == h2.shape[0]-1:
                line += value_format(h2.get_bin_right_edges(1)[-1]) + " ↑"
            if i == 0:
                line += value_format(h2.get_bin_left_edges(1)[0]) + " ↓"
            print(line)
        print("+" + "-" * h2.shape[1] + "+")
        print("←", value_format(h2.get_bin_left_edges(0)[0]))
        colorbar = [
            xtermcolor.colorize("█", bg=0, rgb=(65536 + 256 + 1) * int(j * 255 / (h2.shape[1] + 2)))
            # xtermcolor.colorize(" ", rgb="#000000", bg="#{0}{0}{0}".format(hex(i)[2:].zfill(2)))
            for j in range(h2.shape[1] + 2)
            ]
        colorbar = "".join(colorbar)
        print()
        print("↓", 0)
        print(colorbar)
        print(str(h2.frequencies.max()).rjust(h2.shape[1], " "), "↑")
    types = types + ("map",)
    dims["map"] = [2]
except ImportError:
    pass
