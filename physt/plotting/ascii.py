"""
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