import unittest

import numpy as np

from physt.engines import NumpyEngine, DaskEngine


class EngineTestBase:
    def test_min(self):
        a = np.array([0, 1, 2, -3, 4])
        assert self.engine.min(a) == -3

    def test_max(self):
        a = np.array([0, 1, 2, -3, 4])
        assert self.engine.max(a) == 4

    def test_histogram_with_array(self):
        bins = np.array([0., 1., 2.])
        data = np.array([1.4, 1.2, 0.4, 2.4])
        result = self.engine.histogram(data, bins)
        assert np.array_equal(result, [1, 2])


class TestNumpyEngine(unittest.TestCase, EngineTestBase):
    def setUp(self):
        self.engine = NumpyEngine


class TestDaskEngine(unittest.TestCase, EngineTestBase):
    def setUp(self):
        self.engine = DaskEngine
