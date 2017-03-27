"""Some profile information for h1.

This enabled to reduce the h1 memory footprint between 0.3.25 and 0.3.26 by 60 %.

Usage:
    1) Install memory_profiler (https://pypi.python.org/pypi/memory_profiler)
    2) mprof run profile_h1.py
    3) mprof plot     # See the results
"""
import numpy as np
import physt
import time
import gc


def make_big_h1(data):
    # 0.3.26 The following creates 2 additional copies!!!
    # 0.3.25 The following creates 5 additional copies!!!
    h = physt.h1(data)


def make_big_numpy_hist(data):
    # numpy creates no additional copy (but has no statistics)
    np.histogram(data)


if __name__ == "__main__":
    data = np.random.normal(1, 1, 20000000)
    time.sleep(3)

    make_big_h1(data)
    gc.collect()
    time.sleep(3)

    make_big_numpy_hist(data)
    gc.collect()
    time.sleep(3)
