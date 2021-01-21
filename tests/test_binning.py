from physt import binnings
import numpy as np
import pytest


class TestCalculateBinsNd:
    def test_range(self):
        data1 = np.linspace(0, 10, 100)
        data = np.array([data1, data1]).T
        bins1, bins2 = binnings.calculate_bins_nd(data, range=(4, 5))
        assert bins1.first_edge == 4
        assert bins1.last_edge == 5
        assert bins2.first_edge == 4
        assert bins2.last_edge == 5

    def test_range_partly_none(self):
        data1 = np.linspace(0, 10, 100)
        data = np.array([data1, data1]).T

        bins1, bins2 = binnings.calculate_bins_nd(data, range=((4, 5), None))
        assert bins1.first_edge == 4
        assert bins1.last_edge == 5
        assert bins2.first_edge == 0
        assert bins2.last_edge == 10


class TestNumpyBins:
    def test_int_behaviour(self):
        data = np.random.rand(100)
        the_binning = binnings.numpy_binning(data, 10)
        assert np.allclose(the_binning.numpy_bins, np.histogram(data, 10)[1])

        the_binning = binnings.numpy_binning(data, 10, range=(0.2, 1.0))
        assert np.allclose(the_binning.numpy_bins, np.histogram(data, 10, range=(0.2, 1.0))[1])

    def test_bin_list_behaviour(self):
        data = np.random.rand(100)
        edges = [0.3, 4.5, 5.3, 8.6]
        with pytest.raises(TypeError) as exc:
            the_binning = binnings.numpy_binning(data, edges)


class TestFixedWidthBins:
    def test_without_alignment(self):
        data = np.asarray([4.6, 7.3])
        the_binning = binnings.fixed_width_binning(data, 1.0, align=False)
        assert np.allclose(the_binning.numpy_bins, [4.6, 5.6, 6.6, 7.6])

    def test_with_alignment(self):
        data = np.asarray([4.6, 7.3])
        the_binning = binnings.fixed_width_binning(data, 1.0, align=True)
        assert np.allclose(the_binning.numpy_bins, [4, 5, 6, 7, 8])

    def test_adapt_extension(self):
        b = binnings.FixedWidthBinning(bin_width=10, bin_count=3, min=0, adaptive=True)
        b2 = binnings.FixedWidthBinning(bin_width=10, bin_count=2, min=0, adaptive=True)
        m1, m2 = b2.adapt(b)
        assert tuple(m1) == ((0, 0), (1, 1))
        assert m2 is None
        assert np.array_equal(b2.numpy_bins, [0, 10, 20, 30])
        assert b2.bin_count == 3

    def test_adapt_left(self):
        b = binnings.FixedWidthBinning(bin_width=10, bin_count=3, min=0, adaptive=True)
        b3 = binnings.FixedWidthBinning(bin_width=10, bin_count=2, min=50, adaptive=True)
        m1, m2 = b3.adapt(b)
        assert tuple(m1) == ((0, 5), (1, 6))
        assert tuple(m2) == ((0, 0), (1, 1), (2, 2))
        assert b3.bin_count == 7

    def test_adapt_right(self):
        b = binnings.FixedWidthBinning(bin_width=10, bin_count=3, min=0, adaptive=True)
        b4 = binnings.FixedWidthBinning(bin_width=10, bin_count=2, min=-30, adaptive=True)
        m1, m2 = b4.adapt(b)
        assert tuple(m1) == ((0, 0), (1, 1))
        assert tuple(m2) == ((0, 3), (1, 4), (2, 5))
        assert b4.bin_count == 6

    def test_adapt_intersection1(self):
        b = binnings.FixedWidthBinning(bin_width=10, bin_count=3, min=0, adaptive=True)
        b5 = binnings.FixedWidthBinning(bin_width=10, bin_count=2, min=-10, adaptive=True)
        m1, m2 = b5.adapt(b)
        assert tuple(m1) == ((0, 0), (1, 1))
        assert tuple(m2) == ((0, 1), (1, 2), (2, 3))
        assert b5.bin_count == 4

    def test_adapt_intersection2(self):
        b = binnings.FixedWidthBinning(bin_width=10, bin_count=3, min=0, adaptive=True)
        b6 = binnings.FixedWidthBinning(bin_width=10, bin_count=3, min=10, adaptive=True)
        m1, m2 = b6.adapt(b)
        assert tuple(m1) == ((0, 1), (1, 2), (2, 3))
        assert tuple(m2) == ((0, 0), (1, 1), (2, 2))
        assert b6.bin_count == 4

    def test_adapt_internal(self):
        b1 = binnings.FixedWidthBinning(bin_width=10, bin_count=3, min=0, adaptive=True)
        b2 = binnings.FixedWidthBinning(bin_width=10, bin_count=1, min=10, adaptive=True)
        m1, m2 = b1.adapt(b2)
        assert m1 is None
        assert tuple(m2) == ((0, 1),)

    def test_adapt_external(self):
        b1 = binnings.FixedWidthBinning(bin_width=10, bin_count=1, min=10, adaptive=True)
        b2 = binnings.FixedWidthBinning(bin_width=10, bin_count=3, min=0, adaptive=True)
        m1, m2 = b1.adapt(b2)
        assert tuple(m1) == ((0, 1),)
        assert m2 is None
        assert b1.bin_count == 3

    def test_adapt_wrong(self):
        b1 = binnings.FixedWidthBinning(bin_width=10, bin_count=2, min=0, adaptive=True)
        b2 = binnings.FixedWidthBinning(bin_width=10, bin_count=2, min=1, adaptive=True)
        with pytest.raises(RuntimeError):
            b1.adapt(b2)
        with pytest.raises(RuntimeError):
            b2.adapt(b1)

        b3 = binnings.FixedWidthBinning(bin_width=5, bin_count=6, min=0, adaptive=True)
        with pytest.raises(RuntimeError):
            b1.adapt(b3)
        with pytest.raises(RuntimeError):
            b3.adapt(b1)


class TestHumanBins:
    def test_exact(self):
        data = np.random.rand(1000)
        the_binning = binnings.human_binning(data, 10)
        assert np.allclose(the_binning.numpy_bins, np.linspace(0, 1, 11))

        the_binning = binnings.human_binning(data, 9)
        assert np.allclose(the_binning.numpy_bins, np.linspace(0, 1, 11))

        the_binning = binnings.human_binning(data, 11)
        assert np.allclose(the_binning.numpy_bins, np.linspace(0, 1, 11))

    def test_min_max_bin_width(self):
        data = np.random.rand(1000)

        the_binning = binnings.human_binning(data, min_bin_width=0.3)
        assert the_binning.bin_width == 0.3
        
        the_binning = binnings.human_binning(data, max_bin_width=0.001)
        assert the_binning.bin_width == 0.001              


class TestIntegerBins:
    def test_dice(self):
        data = np.asarray([1, 2, 3, 5, 6, 2, 4, 3, 2, 3, 4, 5, 6, 6, 1, 2, 5])
        the_binning = binnings.integer_binning(data)
        assert np.allclose(the_binning.numpy_bins, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

        the_binning = binnings.integer_binning(data, range=(1, 6))
        assert np.allclose(the_binning.numpy_bins, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])


class TestExponentialBins:
    def test_data(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.5, 3.5, 10.0])
        the_binning = binnings.exponential_binning(data, 2)
        assert np.allclose(the_binning.numpy_bins, [0.1, 1.0, 10.0])

        the_binning = binnings.exponential_binning(data, 2, range=(1.0, 100.0))
        assert np.allclose(the_binning.numpy_bins, [1.0, 10.0, 100.0])


class TestQuantileBins:
    def test_simple(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])
        the_binning = binnings.quantile_binning(data, bin_count=2)
        assert np.allclose(the_binning.numpy_bins, [0.1, 1.0, 10.0])

        the_binning = binnings.quantile_binning(data, bin_count=3)
        assert np.allclose(the_binning.numpy_bins, [0.1, 0.6, 2.2, 10.0])

    def test_qrange(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])
        the_binning = binnings.quantile_binning(data, bin_count=3, qrange=(0.4, 1.0))
        assert np.allclose(the_binning.numpy_bins, [0.76, 1.8, 2.96, 10.])

    def test_q(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])
        the_binning = binnings.quantile_binning(data, q=[0.1, 0.7, 1.0])
        assert np.allclose(the_binning.numpy_bins, [0.26, 2.36, 10.0])
        # TODO: Implement


# TODO: Test binning equality
