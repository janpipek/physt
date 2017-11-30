from typing import Iterable, Tuple

import numpy as np

from .schema import Schema
from .histogram import Histogram


class HistogramBuilder:
    def __init__(self, schemas: Iterable[Schema]):
        self._schemas = tuple(schemas)

    @property
    def schemas(self) -> Tuple[Schema]:
        return self._schemas

    def __call__(self, data) -> Histogram:
        histogram = Histogram()
        histogram.schemas = [schema.copy() for schema in self.schemas]
        if len(self.schemas) == 1:
            values = self.schemas[0].fit_and_apply(data) # TODO: and weights
        else:
            for i, schema in enumerate(self.schemas):
                schema.fit(data[i])
            values = self.multi_apply_schemas(data)
        histogram.values = values
        return histogram

    def multi_apply_schemas(self, data) -> np.ndarray:
        # TODO: Perhaps move to schemas themselves somehow
        edges = [schema.edges for schema in self.schemas]
        masks = [schema.mask for schema in self.schemas]