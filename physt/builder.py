from typing import Iterable, Tuple

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
            data = self.schemas[0].fit_and_apply(data) # TODO: and weights
        else:
            for i, schema in enumerate(self.schemas):
                schema.fit(data[i])
            self.multi_apply_schemas()
        return histogram

    def multi_apply_schemas(self):
        pass