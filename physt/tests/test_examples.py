import unittest
import numpy as np

np.random.seed(42)


class TestExample(unittest.TestCase):
    def test_normal_h1(self):
        from physt.examples import normal_h1
        hist = normal_h1(1000)
        assert hist.ndim == 1
        assert hist.total == 1000

    def test_normal_h2(self):
        from physt.examples import normal_h2
        hist = normal_h2(1000)
        assert hist.ndim == 2
        assert hist.total == 1000