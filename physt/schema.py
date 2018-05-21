"""Bin schemas for physt."""

import math
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np


class Schema:
    """Base class for all one-dimensional schemas.

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
    registered_schemas = OrderedDict()

    def copy(self) -> 'Schema':
        raise NotImplementedError()

    @staticmethod
    def register(name):
        def _decorator(klass: type) -> type:
            Schema.registered_schemas[name] = klass
            return klass
        return _decorator

    @property
    def bins(self) -> np.ndarray:
        edges = self.edges
        bins = np.asarray([edges[:-1], edges[1:]]).T
        mask = self.mask
        if mask is not None:
            bins = bins[mask]
        return bins

    @property
    def edges(self):
        return getattr(self, "_edges", None)

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return self.bins.shape[0]

    @property
    def mask(self):
        return getattr(self, "_mask", None)

    @property
    def bins_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.bins, self.mask

    def fit(self, data):
        raise NotImplementedError()

    def fit_and_apply(self, data, weights=None, dropna: bool=True) -> np.ndarray:
        # TODO: Handle data to make them 1D array ?
        data = np.asarray(data)
        if dropna:
            data = data[~np.isnan(data)]
        self.fit(data)
        numpy_result, _ = np.histogram(data, bins=self.edges, weights=weights)
        mask = self.mask
        if mask is not None:
            return numpy_result[mask].copy()
        else:
            return numpy_result


@Schema.register("static")
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


@Schema.register("numpy")
class NumpySchema(Schema):
    """Binning schema mimicking the behaviour of numpy.histogram"""
    def __init__(self, *, bins: Union[str,int]=10, range=None):
        self.bin_arg = bins
        self.range = range

    @property
    def mask(self):
        return None

    @property
    def bins(self):
        return np.asarray([self._edges[:-1], self._edges[1:]])

    def fit(self, data):
        _, self._edges = np.histogram(data, bins=self.bin_arg, range=self.range)

    def fit_and_apply(self, data, weights=None, dropna: bool=True) -> np.ndarray:
        data = np.asarray(data)
        if dropna:
            data = data[~np.isnan(data)]
        numpy_result, self._edges = np.histogram(data, bins=self.bin_arg, weights=weights,
                                                 range=self.range)
        return numpy_result


@Schema.register("fixed_width")
class FixedWidthSchema(Schema):
    def __init__(self, *, bin_width, bin_count=None, bin_times_min=None, bin_shift=None):
        self._bin_width = bin_width
        self._bin_count = bin_count
        self._bin_times_min = bin_times_min
        self._bin_shift = bin_shift

    def fit(self, data):
        data_min, data_max = data.min(), data.max()
        if self._bin_shift is None:
            self._bin_shift = 0.0
        self._bin_times_min = math.floor((data_min - self._bin_shift) / self._bin_width)
        bin_times_max = math.floor((data_max - self._bin_shift) / self._bin_width) + 1
        self._bin_count = bin_times_max - self._bin_times_min

    @property
    def edges(self):
        indices = np.arange(self._bin_count + 1)
        return self._bin_width * (self._bin_times_min + indices) + self._bin_shift


@Schema.register("integer")
class IntegerSchema(FixedWidthSchema):
    def __init__(self):
        super(IntegerSchema, self).__init__(bin_width=1, bin_shift=0.5)

    # @property
    # def bin_labels(self):
    #     return [str(i) for i in range(self.min_, self.max_ + 1)]

    # TODO: Use something like bincount?


@Schema.register("human")
class HumanSchema(FixedWidthSchema):
    def __init__(self, *, bins="auto", range=None):
        super(HumanSchema, self).__init__(bin_width=None)
        self.bin_arg = bins
        # TODO: deal with range (also in FixedWidth)

    def fit(self, data):
        # TODO: automatic
        bin_count = 20

        subscales = np.array([0.5, 1, 2, 2.5, 5, 10])

        # TODO: ideal_bin_count
        # if bin_count is None:
        #     bin_count = ideal_bin_count(data)
        min_ = data.min()
        max_ = data.max()
        bw = (max_ - min_) / bin_count

        power = np.floor(np.log10(bw)).astype(int)
        best_index = np.argmin(np.abs(np.log(subscales * (10.0 ** power) / bw)))
        
        self._bin_width = (10.0 ** power) * subscales[best_index]
        FixedWidthSchema.fit(self, data)




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


def build_schema(kind: Union[str, type, Schema], **kwargs) -> Schema:
    """Helper method to 
    """
    if isinstance(kind, Schema):
        return kind
    elif isinstance(kind, type):
        return type(**kwargs)
    elif isinstance(kind, str):
        if kind not in Schema.registered_schemas:
            raise ValueError("Unknown schema name, available are: {0}".format(", ").join(Schema.registered_schemas.keys()))
        constructor = StaticSchema.registered_schemas[kind]
        return constructor(**kwargs)
    else:
        raise ValueError("Cannot interpret {0} as schema".format(kind))
