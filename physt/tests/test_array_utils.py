import unittest

import numpy as np

from physt import array_utils


class TestMakeBinArray(unittest.TestCase):
    def test_make_from_numpy(self):
        arr = array_utils.make_bin_array([0, 1, 2])
        assert np.array_equal(arr, [[0, 1], [1, 2]])

    def test_idempotent(self):
        arr = array_utils.make_bin_array([[0, 1], [2, 3]])
        assert np.array_equal(arr, [[0, 1], [2, 3]])


class TestBinsToEdges(unittest.TestCase):
    def test_consecutive(self):
        arr = np.array([[0, 1.1], [1.1, 2.1]])
        edges = array_utils.bins_to_edges(arr)
        assert np.array_equal(edges, [0, 1.1, 2.1])

    def test_inconsecutive(self):
        arr = [[0, 1], [2, 3]]
        with self.assertRaises(array_utils.NonConsecutiveBinsError):
            _ = array_utils.bins_to_edges(arr)


class TestBinsToEdgesAndMask(unittest.TestCase):
    def test_numpy_style(self):
        arr = np.array([1, 2, 3.1, 4])
        edges, mask = array_utils.bins_to_edges_and_mask(arr)
        assert np.array_equal(edges, [1, 2, 3.1, 4])
        assert np.array_equal(mask, [0, 1, 2])

    def test_consecutive(self):
        arr = np.array([[0, 1.1], [1.1, 2.1]])
        edges, mask = array_utils.bins_to_edges_and_mask(arr)
        assert np.array_equal(edges, [0, 1.1, 2.1])
        assert np.array_equal(mask, [0, 1])

    def test_unconsecutive(self):
        arr = np.array([[0, 1], [1.1, 2.1]])
        edges, mask = array_utils.bins_to_edges_and_mask(arr)
        assert np.array_equal(edges, [0, 1, 1.1, 2.1])
        assert np.array_equal(mask, [0, 2])

    def test_nonsense(self):
        arr = np.array([[0, 1], [0.1, 2.1]])
        with self.assertRaises(RuntimeError):
            array_utils.bins_to_edges_and_mask(arr)
        arr = np.array([[[0, 1], [0.1, 2.1]], [[0, 1], [0.1, 2.1]]])
        with self.assertRaises(RuntimeError):
            array_utils.bins_to_edges_and_mask(arr)


class TestValidation(unittest.TestCase):
    def test_rising(self):
        valid = [[[1, 2], [2, 3], [3, 4]], [[1, 2], [3, 4], [4, 5]]]
        for sequence in valid:
            assert array_utils.is_rising((np.array(sequence)))

        invalid = [[[2, 2], [2, 3], [3, 4]], [[1, 2], [1.7, 4], [4, 5]],
                   [[1, 2], [3, 4], [2, 3]]]
        for sequence in invalid:
            assert not array_utils.is_rising((np.array(sequence)))

    def test_consecutive(self):
        valid = [
            [[1, 2], [2, 3], [3, 4]],
            [[1, 2], [2, 1.5], [1.5, 0.7]],
            [[1, 2.2], [2.2, 3], [3, 4]],
        ]
        for sequence in valid:
            assert array_utils.is_consecutive((np.array(sequence)))

        invalid = [[[1, 2], [1.8, 3], [3, 4]], [[1, 2], [2.2, 3], [3, 4]]]
        for sequence in invalid:
            assert not array_utils.is_consecutive((np.array(sequence)))