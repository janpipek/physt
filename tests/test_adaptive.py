import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
from physt import h1
import numpy as np
import pytest


class TestAdaptive(object):
    def test_create_empty(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        assert h.bin_count == 0
        assert h.total == 0
        assert np.allclose(h.bin_widths, 10)
        assert np.isnan(h.mean())
        assert np.isnan(h.std())
        assert h.overflow == 0
        assert h.underflow == 0

    def test_fill_empty(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        h.fill(24)
        assert h.bin_count == 1
        assert np.array_equal(h.bin_left_edges, [20])
        assert np.array_equal(h.bin_right_edges, [30])
        assert np.array_equal(h.frequencies, [1])

    def test_fill_non_empty(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        h.fill(4)
        h.fill(-14)
        assert h.bin_count == 3
        assert h.total == 2
        assert np.array_equal(h.bin_left_edges, [-20, -10, 0])
        assert np.array_equal(h.bin_right_edges, [-10, 0, 10])
        assert np.array_equal(h.frequencies, [1, 0, 1])       

        h.fill(-14)
        assert h.bin_count == 3
        assert h.total == 3

        h.fill(14)
        assert h.bin_count == 4
        assert h.total == 4
        assert np.array_equal(h.frequencies, [2, 0, 1, 1])


class TestFillNAdaptive(object):
    def test_empty(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        h.fill_n([4, 5, 11, 12])
        assert np.array_equal(h.bin_left_edges, [0, 10])
        assert h.total == 4
        assert h.mean() == 8.0

    def test_empty_right_edge(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        h.fill_n([4, 4, 10])
        assert np.array_equal(h.bin_left_edges, [0, 10])
        assert h.total == 3
        assert h.mean() == 6.0

    def test_non_empty(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        h.fill_n([4, 5, 11, 12])

        h.fill_n([-13, 120])
        assert h.bin_left_edges[0] == -20
        assert h.bin_left_edges[-1] == 120
        assert h.bin_count == 15

    def test_with_weights(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        h.fill_n([4, 5, 6, 12], [1, 1, 2, 3])
        assert np.array_equal(h.frequencies, [4, 3])
        assert np.array_equal(h.errors2, [6, 9])
        assert np.array_equal(h.numpy_bins, [0, 10, 20])

    def test_with_incorrect_weights(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        with pytest.raises(RuntimeError):
            h.fill_n([0, 1], [2, 3, 4])
        with pytest.raises(RuntimeError):
            h.fill_n([0, 1, 2, 3], [2, 3, 4])

    def test_empty_exact(self):
        h = h1(None, "fixed_width", 10, adaptive=True)
        h.fill_n([10])
        assert np.array_equal(h.bin_left_edges, [10])
        assert np.array_equal(h.frequencies, [1])
        assert h.total == 1


# class TestChangeBins(object):
#     def test_change_bin_simple(self):
#         h = h1(None, "fixed_width", 10, adaptive=True)
#         h.fill_n([10])
#         assert False

class TestAdaptiveArithmetics(object):
    def test_adding_empty(self):
        ha1 = h1(None, "fixed_width", 10, adaptive=True)
        ha1.fill_n(np.random.normal(100, 10, 1000))

        ha2 = h1(None, "fixed_width", 10, adaptive=True)
        ha3 = ha1 + ha2

        assert ha1 == ha3

        ha4 = ha2 + ha1
        assert ha1 == ha4

    def test_adding_full(self):
        ha1 = h1(None, "fixed_width", 10, adaptive=True)
        ha1.fill_n([1, 43, 23])

        ha2 = h1(None, "fixed_width", 10, adaptive=True)
        ha2.fill_n([23, 51])

        ha3 = ha1 + ha2
        ha4 = ha2 + ha1
        assert np.array_equal(ha3.frequencies, [1, 0, 2, 0, 1, 1])
        assert np.array_equal(ha3.numpy_bins, [0, 10, 20, 30, 40, 50, 60])
        assert ha4 == ha3

    def test_multiplication(self):
        ha1 = h1(None, "fixed_width", 10, adaptive=True)
        ha1.fill_n([1, 43, 23])
        ha1 *= 2
        ha1.fill_n([-2])
        assert np.array_equal(ha1.frequencies, [1, 2, 0, 2, 0, 2])


if __name__ == "__main__":
    pytest.main(__file__)