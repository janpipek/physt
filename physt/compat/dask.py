"""Dask-based and dask oriented variants of physt histogram facade functions."""
from typing import TYPE_CHECKING

import dask

from physt import h1 as original_h1
from physt import histogramdd as original_hdd

if TYPE_CHECKING:
    import dask.array

options = {"chunk_split": 16}


def _run_dask(
    *,
    name: str,
    data: "dask.array.Array",
    compute: bool,
    method,
    func,
    expand_arg: bool = False,
):
    """Construct the computation graph and optionally compute it.

    :param name: Name of the method (for graph naming purposes).
    :param data: Dask array data
    :param func: Function running of each array chunk.
    :param compute: If True, compute immediately
    :param method: None (linear execution), "threaded" or callable
        to apply when computing.
    """
    if expand_arg:
        graph = dict(
            ("{0}-{1}-{2}".format(name, data.name, index), (func, *item))
            for index, item in enumerate(data.__dask_keys__())
        )
    else:
        graph = dict(
            ("{0}-{1}-{2}".format(name, data.name, index), (func, item))
            for index, item in enumerate(data.__dask_keys__())
        )
    items = list(graph.keys())
    result_name = "{0}-{1}-result".format(name, data.name)
    graph.update(data.dask)
    graph[result_name] = (sum, items)
    if compute:
        if not method:
            return dask.get(graph, result_name)
        if method in ["thread", "threaded", "threading", "threads"]:
            return dask.threaded.get(graph, result_name)
        return method(graph, result_name)
    return graph, result_name


def histogram1d(data, bins=None, **kwargs):
    """Facade function to create one-dimensional histogram using dask.

    Parameters
    ----------
    data: dask.DaskArray or array-like (can have more than one dimension)

    See also
    --------
    physt.histogram
    """
    if not hasattr(data, "dask"):
        data = dask.array.from_array(data, chunks=int(data.shape[0] / options["chunk_split"]))

    if not kwargs.get("adaptive", True):
        raise RuntimeError("Only adaptive histograms supported for dask (currently).")
    kwargs["adaptive"] = True

    def block_hist(array):
        return original_h1(array, bins, **kwargs)

    return _run_dask(
        name="dask_adaptive1d",
        data=data,
        compute=kwargs.pop("compute", True),
        method=kwargs.pop("dask_method", "threaded"),
        func=block_hist,
    )


h1 = histogram1d  # Alias for convenience


def histogramdd(data, bins=None, **kwargs):
    """Facade function to create multi-dimensional histogram using dask.

    Each "column" must be one-dimensional.
    """
    from dask.array.rechunk import rechunk

    if isinstance(data, (list, tuple)):
        data = dask.array.stack(data, axis=1)

    if not hasattr(data, "dask"):
        data = dask.array.from_array(
            data, chunks=(int(data.shape[0] / options["chunk_split"]), data.shape[1])
        )
    else:
        data = rechunk(data, {1: data.shape[1]})

    if isinstance(data, dask.array.Array):
        if data.ndim != 2:
            raise ValueError(
                f"Only (n, dim) data allowed for histogramdd, {data.shape} encountered."
            )

    if not kwargs.get("adaptive", True):
        raise ValueError("Only adaptive histograms supported for dask (currently).")
    kwargs["adaptive"] = True

    def block_hist(array):
        return original_hdd(array, bins, **kwargs)

    return _run_dask(
        name="dask_adaptive_dd",
        data=data,
        compute=kwargs.pop("compute", True),
        method=kwargs.pop("dask_method", "threaded"),
        func=block_hist,
        expand_arg=True,
    )


def histogram2d(data1, data2, bins=None, **kwargs):
    """Facade function to create 2D histogram using dask."""
    # TODO: currently very unoptimized! for non-dasks
    if "axis_names" not in kwargs:
        if hasattr(data1, "name") and hasattr(data2, "name"):
            kwargs["axis_names"] = [data1.name, data2.name]
    if not hasattr(data1, "dask"):
        data1 = dask.array.from_array(data1, chunks=data1.size() / 100)
    if not hasattr(data2, "dask"):
        data2 = dask.array.from_array(data2, chunks=data2.size() / 100)

    data = dask.array.stack([data1, data2], axis=1)
    kwargs["dim"] = 2
    return histogramdd(data, bins, **kwargs)


h2 = histogram2d  # Alias for convenience


def h3(data, bins=None, **kwargs):
    """Facade function to create 3D histogram using dask."""
    return histogramdd(data, bins, **kwargs)
