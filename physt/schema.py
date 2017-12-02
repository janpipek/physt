import numpy as np
from typing import Tuple, Union


class Schema:
    """

    Attributes:
        - fitted [bad name - sklearn?]
        - adaptive [bad name]
        - bins
        - edges
        - mask

    Methods:
        - fit
        - apply
        - fit_and_apply
    """
    def copy(self) -> 'Schema':
        raise NotImplementedError()

    @property
    def bins(self) -> np.ndarray:
        if hasattr(self, "_bins") and self._bins is not None:
            return self._bins
        else:
            return

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        if hasattr(self, "_bins") and self._bins is not None:
            return (self._bins.shape[0],)
        elif hasattr(self, "_edges") and self._edges is not None:
            return len(self._edges) - 1,
        else:
            return 0

    @property
    def edges(self):
        return self._edges

    @property
    def mask(self):
        return self._mask

    @property
    def bins_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        return

    def fit(self, data):
        raise NotImplementedError()

    def fit_and_apply(self, data, weights=None) -> np.ndarray:
        # TODO: Handle data to make them 1D array ?
        data = np.asarray(data)
        self.fit(data)
        numpy_result, _ = np.histogram(data, bins=self.edges, weights=weights)
        mask = self.mask
        if mask:
            return numpy_result[mask].copy()
        else:
            return numpy_result


class StaticSchema(Schema):
    def __init__(self, *, bins=None, edges=None, mask=None):
        if bins is not None:
            if edges is not None:
                raise ValueError("Cannot specify both bins and edges at the same time.")
            self._bins = bins.copy()
        elif edges is not None:
            self._edges = edges.copy()
        else:
            raise ValueError("Must specify either bins or edges.")
        self._mask = mask

    def fit(self, data):
        pass

    def copy(self) -> 'StaticSchema':
        return self.__class__(bins=self._bins, edges=self._edges, mask=self._mask)


class NumpySchema(Schema):
    """Binning schema mimicking the behaviour of numpy.histogram"""
    def __init__(self, bins: Union[str,int]=10, range=None):
        self.bin_arg = bins
        self.range = range

    @property
    def mask(self):
        return None

    @property
    def bins(self):
        pass
        # TODO: Combine fields

    @property
    def edges(self):
        return self._edges

    def fit(self, data):
        _, self._edges = np.histogram(data, bins=self.bin_arg, range=self.range)

    def fit_and_apply(self, data, weights=None) -> np.ndarray:
        numpy_result, self._edges = np.histogram(data, bins=self.bin_arg, weights=weights,
                                                 range=self.range)
        return numpy_result


class IntegerSchema(Schema):
    pass
    # TODO: Use something like bincount?


class MultiSchema:
    def __init__(self, schemas):
        self._schemas = tuple(schemas)

    @property
    def ndim(self):
        return len(self._schemas)

    @property
    def shape(self):
        result = ()
        for schema in self._schemas:
            result += schema.shape
        return result

    @property
    def edges(self):
        return [schema.edges for schema in self.schemas]

    @property
    def schemas(self):
        return self._schemas

    def __getitem__(self, item):
        return self._schemas[item]

    def fit(self, data):
        # TODO: data size check
        for i, schema in enumerate(self.schemas):
            schema.fit(data[:,i])

    def fit_and_apply(self, data, weights=None) -> np.ndarray:
        self.fit(data)
        return self.apply(data, weights=weights)

    def apply(self, data, weights=None) -> np.ndarray:
        edges = [schema.edges for schema in self.schemas]
        masks = [schema.mask for schema in self.schemas]
        values, _ = np.histogramdd(data, bins=edges, weights=weights)

        changed = False
        for i, mask in enumerate(masks):
            if mask:
                # TODO: Apply the subselection
                pass

        if changed:
            values = values.copy()
        return values


def build_schema(kind="human", *args, **kwargs) -> Schema:
    """
    """
