import numpy as np


try:
    import dask
    import dask.array as da
    DASK_ENABLED = True
except ImportError:
    DASK_ENABLED = False


class Engine:
    @classmethod
    def get_range(cls, a, **kwargs):
        if kwargs.get("range", None):
            return kwargs["range"]
        return cls.min(a), cls.max(a)

    @classmethod
    def min(cls, a):
        return np.min(a)

    @classmethod
    def max(cls, a):
        return np.max(a)

    @classmethod
    def apply(cls, a, bins):
        pass


class NumpyEngine(Engine):
    @classmethod
    def histogram(cls, a, bins, weights=None):
        import numpy as np
        return np.histogram(a, bins=bins, weights=weights)[0]

    @classmethod
    def prepare_data(cls, a):
        return np.asarray(a)

    # TODO: histogramdd


class DaskEngine(Engine):
    fallback = NumpyEngine

    chunk_split = 16

    @classmethod
    def _run_dask(cls, name, data, compute, method, func):
        import dask
        data_hash = str(id(data))[-6:]
        graph = dict(("{0}-{1}-{2}".format(name, data_hash, i), (func, k))
                    for i, k in enumerate(dask.core.flatten(data.__dask_keys__())))
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

    @classmethod
    def prepare_data(cls, a):
        if hasattr(a, "dask"):
            return a
        else:
            a = np.asarray(a).flatten()
            print(a.shape)
            print(int(a.shape[0] / cls.chunk_split))
            return da.from_array(a, chunks=max(1, int(a.shape[0] / cls.chunk_split)))

    @classmethod
    def histogram(cls, a, bins, weights=None, **kwargs):
        # Idempotent
        import dask.array as da

        a = cls.prepare_data(a)
        weights = cls.prepare_data(weights) if weights is not None else None
        values, _ = da.histogram(a, bins=bins, weights=weights)
        return values.compute()

        #def block_hist(array):
        #    return NumpyEngine.histogram(array, bins=bins)

        #return cls._run_dask(
        #    name="dask_histogram1d",
        #    data=a,
        #    compute=True,   # TODO: Can we not compute?
        #    method=kwargs.pop("dask_method", "threaded"),
        #    func=block_hist)

    # TODO: histogramdd


class TensorflowEagerEngine(Engine):
    fallback = NumpyEngine


class TensorflowEngine(Engine):
    fallback = NumpyEngine


class PytorchEngine(Engine):
    fallback = NumpyEngine


def find_best_engine(a):
    if DASK_ENABLED:
        return DaskEngine
    else:
        return NumpyEngine