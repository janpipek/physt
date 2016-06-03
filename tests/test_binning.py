import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
from physt import binning
import numpy as np
import pytest

# TODO: Enable in Python < 3.3
#


class TestNumpyBins(object):
    def test_int_behaviour(self):
        data = np.random.rand(100)
        bins = binning.numpy_bins(data, 10)
        assert np.allclose(bins, np.histogram(data, 10)[1])

        bins = binning.numpy_bins(data, 10, range=(0.2, 1.0))
        assert np.allclose(bins, np.histogram(data, 10, range=(0.2, 1.0))[1])

    def test_bin_list_behaviour(self):
        data = np.random.rand(100)
        edges = [0.3, 4.5, 5.3, 8.6]
        bins = binning.numpy_bins(data, edges)
        assert np.allclose(bins, edges)
        assert np.allclose(bins, np.histogram(data, edges)[1])


class TestFixedWidthBins(object):
    def test_without_alignment(self):
        data = np.asarray([4.6, 7.3])
        bins = binning.fixed_width_bins(data, 1.0, False)
        assert np.allclose(bins, [4.6, 5.6, 6.6, 7.6])

    def test_with_alignment(self):
        data = np.asarray([4.6, 7.3])
        bins = binning.fixed_width_bins(data, 1.0, True)
        assert np.allclose(bins, [4, 5, 6, 7, 8])

        bins = binning.fixed_width_bins(data, 1.0, align=0.5)
        assert np.allclose(bins, [4.5, 5.5, 6.5, 7.5])

        bins = binning.fixed_width_bins(data, 1.0, align=1.5)
        assert np.allclose(bins, [4.5, 5.5, 6.5, 7.5])

        bins = binning.fixed_width_bins(data, 1.0, align=3.0)
        assert np.allclose(bins, [3, 4, 5, 6, 7, 8, 9])

        bins = binning.fixed_width_bins(data, 1.0, align=10)
        assert np.allclose(bins, np.arange(0, 11))                


class TestHumanBins(object):
    def test_exact(self):
        data = np.random.rand(1000)
        bins = binning.human_bins(data, 10)
        assert np.allclose(bins, np.linspace(0, 1, 11))

        bins = binning.human_bins(data, 9)
        assert np.allclose(bins, np.linspace(0, 1, 11))

        bins = binning.human_bins(data, 11)
        assert np.allclose(bins, np.linspace(0, 1, 11))


class TestIntegerBins(object):
    def test_dice(self):
        data = np.asarray([1, 2, 3, 5, 6, 2, 4, 3, 2, 3, 4, 5, 6, 6, 1, 2, 5])
        bins = binning.integer_bins(data)
        assert np.allclose(bins, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

        bins = binning.integer_bins(data, range=(1, 6))
        assert np.allclose(bins, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])


class TestExponentialBins(object):
    def test_data(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.5, 3.5, 10.0])
        bins = binning.exponential_bins(data, 2)
        assert np.allclose(bins, [0.1, 1.0, 10.0])

        bins = binning.exponential_bins(data, 2, range=(1.0, 100.0))
        assert np.allclose(bins, [1.0, 10.0, 100.0])        


class TestQuantileBins(object):
    def test_simple(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])
        bins = binning.quantile_bins(data, 2)
        assert np.allclose(bins, [0.1, 1.0, 10.0])

        bins = binning.quantile_bins(data, 3)
        assert np.allclose(bins, [0.1, 0.6, 2.2, 10.0])

    def test_qrange(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])
        bins = binning.quantile_bins(data, 3, qrange=(0.4, 1.0))
        assert np.allclose(bins, [0.76, 1.8, 2.96, 10.])


if sys.version_info >= (3, 3):
    from unittest import mock
else:
    try:
        import mock
    except:
        raise RuntimeError("You need to have 'mock' package installed in Python < 3.3")


class TestCalculateBins(object):
    def test_proper_forwarding(self):
        for key in list(binning.binning_methods.keys()):
            binning.binning_methods[key] = mock.MagicMock()
        array = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])

        for key in list(binning.binning_methods.keys()):
            binning.calculate_bins(array, key)
            binning.binning_methods[key].assert_called_once_with(array)

    def test_implicit_numpy_calls(self):
        # TODO: Write this
        pass
