from __future__ import absolute_import
from . import h1 as original_h1
from . import histogramdd as original_hdd


options = {
    "chunk_split": 16
}


def _run_dask(name, data, compute, method, func):
    import dask
    data_hash = str(id(data))[-6:]
    graph = dict(("{0}-{1}-{2}".format(name, data_hash, i), (func, k))
                    for i, k in enumerate(dask.core.flatten(data._keys())))
    items = list(graph.keys())
    graph.update(data.dask)   
    result_name = "{0}-{1}-result".format(name, data_hash)
    graph[result_name] = (sum, items)
    if compute:
        if not method:
            return dask.get(graph, result_name)
        elif method in ["thread", "threaded", "threading", "threads"]:
            return dask.threaded.get(graph, result_name)
        else:
            return method(graph, result_name)
    else:
        return graph, result_name


def histogram1d(data, bins=None, *args, **kwargs):
    import dask
    if not hasattr(data, "dask"):
        data = dask.array.from_array(data, chunks=int(data.shape[0] / options["chunk_split"]))

    if not kwargs.get("adaptive", True):
        raise RuntimeError("Only adaptive histograms supported for dask (currently).")
    kwargs["adaptive"] = True
    
    def block_hist(array):
        return original_h1(array, bins, *args, **kwargs)
    
    return _run_dask(
        name="dask_adaptive1d",
        data=data,
        compute=kwargs.pop("compute", True),
        method=kwargs.pop("dask_method", "threaded"),
        func=block_hist)


h1 = histogram1d


def histogramdd(data, bins=None, *args, **kwargs):
    import dask
    from dask.array.rechunk import rechunk
    if not hasattr(data, "dask"):
        data = dask.array.from_array(data, chunks=(int(data.shape[0] / options["chunk_split"]), data.shape[1]))
    else:
        data = rechunk(data, {1: data.shape[1]})

    name = "dask_adaptive"
    if not kwargs.get("adaptive", True):
        raise RuntimeError("Only adaptive histograms supported for dask (currently).")
    kwargs["adaptive"] = True
    def block_hist(array):
        return original_hdd(array, bins, *args, **kwargs)
  
    return _run_dask(
        name="dask_adaptive2d",
        data=data,
        compute=kwargs.pop("compute", True),
        method=kwargs.pop("dask_method", "threaded"),
        func=block_hist)


def histogram2d(data1, data2, bins=None, *args, **kwargs):
    import dask
    if not "axis_names" in kwargs:
        if hasattr(data1, "name") and hasattr(data2, "name"):
            kwargs["axis_names"] = [data1.name, data2.name]    
    if not hasattr(data1, "dask"):
        data1 = dask.array.from_array(data1, chunks=data1.size() / 100)
    if not hasattr(data2, "dask"):
        data2 = dask.array.from_array(data2, chunks=data2.size() / 100)

    data = dask.array.stack([data1, data2], axis=1)
    kwargs["dim"] = 2
    return histogramdd(data, bins, *args, **kwargs) 


h2 = histogram2d
def h3(data, *args, **kwargs):
    return histogramdd(data, dim=3, *args, **kwargs)