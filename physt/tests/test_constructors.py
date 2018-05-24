import unittest

from physt.constructors import h1
from physt.histogram import Histogram


class TestH1(unittest.TestCase):
    def test_no_params_no_data(self):
        from physt.schema import HumanSchema

        histogram = h1()
        assert isinstance(histogram, Histogram)
        assert isinstance(histogram.schema, HumanSchema)
        assert histogram.values is None
        assert histogram.bins is None
        