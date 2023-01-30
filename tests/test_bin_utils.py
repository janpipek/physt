import numpy as np
import pytest

from physt._bin_utils import (
    find_human_width,
    is_consecutive,
    is_rising,
    make_bin_array,
    to_numpy_bins_with_mask,
)


class TestMakeArray:
    def test_make_from_numpy(self):
        arr = make_bin_array([0, 1, 2])
        assert np.array_equal(arr, [[0, 1], [1, 2]])

    def test_idempotent(self):
        arr = make_bin_array([[0, 1], [2, 3]])
        assert np.array_equal(arr, [[0, 1], [2, 3]])


class TestNumpyBinsWithMask:
    def test_numpy_style(self):
        arr = np.array([1, 2, 3.1, 4])
        edges, mask = to_numpy_bins_with_mask(arr)
        assert np.array_equal(edges, [1, 2, 3.1, 4])
        assert np.array_equal(mask, [0, 1, 2])

    def test_consecutive(self):
        arr = np.array([[0, 1.1], [1.1, 2.1]])
        edges, mask = to_numpy_bins_with_mask(arr)
        assert np.array_equal(edges, [0, 1.1, 2.1])
        assert np.array_equal(mask, [0, 1])

    def test_unconsecutive(self):
        arr = np.array([[0, 1], [1.1, 2.1]])
        edges, mask = to_numpy_bins_with_mask(arr)
        assert np.array_equal(edges, [0, 1, 1.1, 2.1])
        assert np.array_equal(mask, [0, 2])

    def test_nonsense(self):
        arr = np.array([[0, 1], [0.1, 2.1]])
        with pytest.raises(ValueError, match="to_numpy_bins_with_mask: edges array not monotone"):
            to_numpy_bins_with_mask(arr)
        arr = np.array([[[0, 1], [0.1, 2.1]], [[0, 1], [0.1, 2.1]]])
        with pytest.raises(
            ValueError, match="to_numpy_bins_with_mask: array with dim=1 or 2 expected"
        ):
            to_numpy_bins_with_mask(arr)


class TestValidation:
    def test_rising(self):
        valid = [[[1, 2], [2, 3], [3, 4]], [[1, 2], [3, 4], [4, 5]]]
        for sequence in valid:
            assert is_rising((np.array(sequence)))

        invalid = [
            [[2, 2], [2, 3], [3, 4]],
            [[1, 2], [1.7, 4], [4, 5]],
            [[1, 2], [3, 4], [2, 3]],
        ]
        for sequence in invalid:
            assert not is_rising((np.array(sequence)))

    def test_consecutive(self):
        valid = [
            [[1, 2], [2, 3], [3, 4]],
            [[1, 2], [2, 1.5], [1.5, 0.7]],
            [[1, 2.2], [2.2, 3], [3, 4]],
        ]
        for sequence in valid:
            assert is_consecutive((np.array(sequence)))

        invalid = [[[1, 2], [1.8, 3], [3, 4]], [[1, 2], [2.2, 3], [3, 4]]]
        for sequence in invalid:
            assert not is_consecutive((np.array(sequence)))


class TestFindHumanWidth:
    def test_normal(self):
        invalues = [1.1, 2.4, 32, 57, 2000, 3621, 85000]
        expected = [1.0, 2.5, 25, 50, 2000, 5000, 1e5]
        result = [find_human_width(x) for x in invalues]
        assert np.array_equal(result, expected)

    def test_time(self):
        invalues = [1.1, 2.4, 32, 57, 2000, 3621, 85000]
        expected = np.asarray([1.0, 2.0, 30, 60, 1800, 3600, 86400], dtype=float)
        result = [find_human_width(x, kind="time") for x in invalues]
        assert np.array_equal(result, expected)
