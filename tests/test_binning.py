import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
from physt import binnings
import numpy as np
import pytest

# TODO: Enable in Python < 3.3
#


class TestNumpyBins(object):
    def test_int_behaviour(self):
        data = np.random.rand(100)
        the_binning = binnings.numpy_binning(data, 10)
        assert np.allclose(the_binning.numpy_bins, np.histogram(data, 10)[1])

        the_binning = binnings.numpy_binning(data, 10, range=(0.2, 1.0))
        assert np.allclose(the_binning.numpy_bins, np.histogram(data, 10, range=(0.2, 1.0))[1])

    def test_bin_list_behaviour(self):
        data = np.random.rand(100)
        edges = [0.3, 4.5, 5.3, 8.6]
        the_binning = binnings.numpy_binning(data, edges)
        assert np.allclose(the_binning.numpy_bins, edges)
        assert np.allclose(the_binning.numpy_bins, np.histogram(data, edges)[1])


class TestFixedWidthBins(object):
    def test_without_alignment(self):
        data = np.asarray([4.6, 7.3])
        the_binning = binnings.fixed_width_binning(data, 1.0, align=False)
        assert np.allclose(the_binning.numpy_bins, [4.6, 5.6, 6.6, 7.6])

    def test_with_alignment(self):
        data = np.asarray([4.6, 7.3])
        the_binning = binnings.fixed_width_binning(data, 1.0, align=True)
        assert np.allclose(the_binning.numpy_bins, [4, 5, 6, 7, 8])


class TestHumanBins(object):
    def test_exact(self):
        data = np.random.rand(1000)
        the_binning = binnings.human_binning(data, 10)
        assert np.allclose(the_binning.numpy_bins, np.linspace(0, 1, 11))

        the_binning = binnings.human_binning(data, 9)
        assert np.allclose(the_binning.numpy_bins, np.linspace(0, 1, 11))

        the_binning = binnings.human_binning(data, 11)
        assert np.allclose(the_binning.numpy_bins, np.linspace(0, 1, 11))


class TestIntegerBins(object):
    def test_dice(self):
        data = np.asarray([1, 2, 3, 5, 6, 2, 4, 3, 2, 3, 4, 5, 6, 6, 1, 2, 5])
        the_binning = binnings.integer_binning(data)
        assert np.allclose(the_binning.numpy_bins, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

        the_binning = binnings.integer_binning(data, range=(1, 6))
        assert np.allclose(the_binning.numpy_bins, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])


class TestExponentialBins(object):
    def test_data(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.5, 3.5, 10.0])
        the_binning = binnings.exponential_binning(data, 2)
        assert np.allclose(the_binning.numpy_bins, [0.1, 1.0, 10.0])

        the_binning = binnings.exponential_binning(data, 2, range=(1.0, 100.0))
        assert np.allclose(the_binning.numpy_bins, [1.0, 10.0, 100.0])        


class TestQuantileBins(object):
    def test_simple(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])
        the_binning = binnings.quantile_binning(data, 2)
        assert np.allclose(the_binning.numpy_bins, [0.1, 1.0, 10.0])

        the_binning = binnings.quantile_binning(data, 3)
        assert np.allclose(the_binning.numpy_bins, [0.1, 0.6, 2.2, 10.0])

    def test_qrange(self):
        data = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])
        the_binning = binnings.quantile_binning(data, 3, qrange=(0.4, 1.0))
        assert np.allclose(the_binning.numpy_bins, [0.76, 1.8, 2.96, 10.])


        # TODO: Rework the binning
# if sys.version_info >= (3, 3):
#     from unittest import mock
# else:
#     try:
#         import mock
#     except:
#         raise RuntimeError("You need to have 'mock' package installed in Python < 3.3")
#
#
# class TestCalculateBins(object):
#     def test_proper_forwarding(self):
#         for key in list(binning.binning_methods.keys()):
#             binning.binning_methods[key] = mock.MagicMock()
#         array = np.asarray([0.1, 0.3, 0.4, 0.7, 1.0, 2.0, 2.6, 3.5, 10.0])
#
#         for key in list(binning.binning_methods.keys()):
#             binning.calculate_bins(array, key)
#             binning.binning_methods[key].assert_called_once_with(array)
#
#     def test_implicit_numpy_calls(self):
#         # TODO: Write this
#         pass
