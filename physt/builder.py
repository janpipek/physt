from typing import Iterable, Tuple

import numpy as np

from .schema import Schema, multi_apply_schemas
from .histogram import Histogram


class HistogramBuilder:
    def __init__(self, schemas: Iterable[Schema]):
        self._schemas = tuple(schemas)

    @property
    def schemas(self) -> Tuple[Schema]:
        return self._schemas

    def fit(self, data):
        for i, schema in enumerate(self.schemas):
            schema.fit(data[i])

    def apply(self, data, weights=None) -> Histogram:
        histogram = Histogram()
        histogram.schemas = [schema.copy() for schema in self.schemas]
        if len(self.schemas) == 1:
            values = self.schemas[0].apply(data, weights)
        else:
            for i, schema in enumerate(self.schemas):
                schema.fit(data[i])
            values = multi_apply_schemas(data, weights)
        histogram.values = values
        return histogram

