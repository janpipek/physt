import numpy as np
import pytest

from physt.histogram1d import Histogram1D
from physt import h1, h2, histogramdd


@pytest.fixture
def empty_adaptive1(create_adaptive) -> Histogram1D:
    """One-dimesion adaptive fixed-width histogram with bin_width=1."""
    return create_adaptive((0,))


class TestAdaptive(object):
    def test_create_empty(self):
        h = h1(None, "fixed_width", bin_width=10, adaptive=True)
        assert h.bin_count == 0
        assert h.total == 0
        assert np.allclose(h.bin_widths, 10)
        assert np.isnan(h.mean())
        assert np.isnan(h.std())
        assert h.overflow == 0
        assert h.underflow == 0

    def test_fill_empty(self, create_adaptive):
        h = create_adaptive((0,))
        h.fill(2.4)
        assert h.bin_count == 1
        assert np.array_equal(h.bin_left_edges, [2])
        assert np.array_equal(h.bin_right_edges, [3])
        assert np.array_equal(h.frequencies, [1])

    def test_fill_non_empty(self, create_adaptive):
        h = create_adaptive((3,))
        h.fill(3.7)
        assert h.bin_count == 4
        assert h.total == 4
        assert np.array_equal(h.bin_left_edges, [0, 1, 2, 3])
        assert np.array_equal(h.bin_right_edges, [1, 2, 3, 4])
        assert np.array_equal(h.frequencies, [0, 1, 2, 1])

        h.fill(-0.7)
        assert h.bin_count == 5
        assert h.total == 5

        h.fill(10.1)
        assert h.bin_count == 12
        assert h.total == 6
        assert np.array_equal(h.frequencies, [1, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1])


class TestFillNAdaptive(object):
    def test_empty(self, empty_adaptive1):
        h = empty_adaptive1
        h.fill_n([1.2, 2.3, 2.5, 2.6])
        assert np.array_equal(h.bin_left_edges, [1, 2])
        assert h.total == 4
        assert np.array_equal(h.frequencies, [1, 3])

    def test_empty_right_edge(self, empty_adaptive1):
        h = empty_adaptive1
        h.fill_n([0.4, 0.4, 1.0])
        assert np.array_equal(h.bin_left_edges, [0, 1])
        assert h.total == 3
        assert np.array_equal(h.frequencies, [2, 1])

    def test_non_empty(self, create_adaptive):
        h = create_adaptive((3,))

        h.fill_n([-1.3, 4.1])
        assert h.bin_left_edges[0] == -2.0
        assert h.bin_left_edges[-1] == 4.0
        assert h.bin_count == 7
        assert h.total == 5

    def test_with_weights(self, empty_adaptive1):
        h = empty_adaptive1
        h.fill_n([.4, .5, .6, 1.2], [1, 1, 2, 3])
        assert np.array_equal(h.frequencies, [4, 3])
        assert np.array_equal(h.errors2, [6, 9])
        assert np.array_equal(h.numpy_bins, [0, 1.0, 2.0])

    def test_with_incorrect_weights(self, empty_adaptive1):
        h = empty_adaptive1
        with pytest.raises(ValueError):
            h.fill_n([0, 1], [2, 3, 4])
        with pytest.raises(ValueError):
            h.fill_n([0, 1, 2, 3], [2, 3, 4])

    def test_empty_exact(self, empty_adaptive1):
        h = empty_adaptive1
        h.fill_n([1.0])
        assert np.array_equal(h.bin_left_edges, [1.0])
        assert np.array_equal(h.frequencies, [1])
        assert h.total == 1


class TestAdaptive2D(object):
    def test_create_empty(self):
        h = h2(None, None, "fixed_width", bin_width=10, adaptive=True)
        for b in h._binnings:
            assert b.is_adaptive()
        assert h.ndim == 2

    def test_create_nonempty(self):
        d1 = [1, 21, 3]
        d2 = [11, 12, 13]
        h = h2(d1, d2, "fixed_width", bin_width=10, adaptive=True)
        assert h.shape == (3, 1)

    def test_fill_empty(self, create_adaptive):
        h = create_adaptive((0, 0))
        h.fill([1.2, 1.3])
        assert h.total == 1
        assert np.array_equal(h.numpy_bins, [[1, 2], [1, 2]])

    def test_fill_nonempty(self, create_adaptive):
        h = create_adaptive((2, 2))
        h.fill([2.1, 1.8])
        assert h.total == 7
        assert np.array_equal(h.numpy_bins[0], [0, 1, 2, 3])
        assert np.array_equal(h.numpy_bins[1], [0, 1, 2])


class TestAdaptiveND:
    def test_create_empty(self):
        h = histogramdd(None, "fixed_width", bin_width=10, dim=7, adaptive=True)
        assert h.ndim == 7
        assert h.is_adaptive()

    # TODO: Add a few more tests?


class TestAdaptiveArithmetics:
    def test_adding_empty(self, create_adaptive, empty_adaptive1):
        h = create_adaptive((5,))
        assert h + empty_adaptive1 == h
        assert empty_adaptive1 + h == h

    def test_adding_full(self):
        ha1 = h1(None, "fixed_width", bin_width=10, adaptive=True)
        ha1.fill_n([1, 43, 23])

        ha2 = h1(None, "fixed_width", bin_width=10, adaptive=True)
        ha2.fill_n([23, 51])

        ha3 = ha1 + ha2
        ha4 = ha2 + ha1
        assert np.array_equal(ha3.frequencies, [1, 0, 2, 0, 1, 1])
        assert np.array_equal(ha3.numpy_bins, [0, 10, 20, 30, 40, 50, 60])
        assert ha4 == ha3

    def test_add_cross_2d(self):
        d1 = np.asarray([1, 21, 3])
        d2 = np.asarray([10, 12, 20])
        h = h2(d1, d2, "fixed_width", bin_width=10, adaptive=True)
        hb = h2(d1, d2 +10, "fixed_width", bin_width=10, adaptive=True)
        hc = h + hb
        assert np.array_equal(hc.numpy_bins[1], [10, 20, 30, 40])
