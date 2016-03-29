import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from physt.histogram1d import Histogram1D
import numpy as np
import pytest

bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
values = [4, 0, 3, 7.2]
example = Histogram1D(bins, values, overflow=1, underflow=2)


class TestBins(object):
    def test_nbins(self):
        assert example.bin_count == 4

    def test_edges(self):
        assert np.allclose(example.bin_left_edges, [1.2, 1.4, 1.5, 1.7])
        assert np.allclose(example.bin_right_edges, [1.4, 1.5, 1.7, 1.8])
        assert np.allclose(example.bin_centers, [1.3, 1.45, 1.6, 1.75])

    def test_numpy_bins(self):
        assert np.allclose(example.numpy_bins, [1.2, 1.4, 1.5, 1.7, 1.8 ])

    def test_widths(self):
        assert np.allclose(example.bin_widths, [0.2, 0.1, 0.2, 0.1])


class TestValues(object):
    def test_values(self):
        assert np.allclose(example.frequencies, [4, 0, 3, 7.2])

    def test_cumulative_values(self):
        assert np.allclose(example.cumulative_frequencies, [4, 4, 7, 14.2])

    def test_normalize(self):
        new = example.normalize()
        expected = np.array([4, 0, 3, 7.2]) / 17.2
        assert np.allclose(new.frequencies, expected)
        assert np.array_equal(new.bins, example.bins)
        assert new is not example

        copy = example.copy()
        new = copy.normalize(inplace=True)
        assert np.allclose(new.frequencies, expected)
        assert np.array_equal(new.bins, example.bins)
        assert new is copy

    def test_total(self):
        assert np.isclose(example.total, 17.2)


class TestCopy(object):
    def test_copy(self):
        new = example.copy()
        assert new is not example
        assert new.bins is not example.bins
        assert new.frequencies is not example.frequencies
        assert new == example


class TestEquivalence(object):
    def test_eq(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [4, 0, 3, 7.2]
        other1 = Histogram1D(bins, values, underflow=2, overflow=1)
        assert other1 == example

        bins = [1.22, 1.4, 1.5, 1.7, 1.8 ]
        values = [4, 0, 3, 7.2]
        other2 = Histogram1D(bins, values, underflow=2, overflow=1)
        assert other2 != example

        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [4, 0, 13, 7.2]
        other3 = Histogram1D(bins, values, underflow=2, overflow=1)
        assert other3 != example

    def test_eq_with_underflows(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [4, 0, 3, 7.2]
        other1 = Histogram1D(bins, values, underflow=2)
        assert other1 != example

        other2 = Histogram1D(bins, values, overflow=1)
        assert other2 != example

        other3 = Histogram1D(bins, values, overflow=1, underflow=2)
        assert other3 == example


class TestIndexing(object):
    def test_single(self):
        zeroth = example[0]
        assert np.allclose(zeroth[0], (1.2, 1.4))
        assert zeroth[1] == 4

        other = example[-2]
        assert np.allclose(other[0], (1.5, 1.7))
        assert other[1] == 3

        with pytest.raises(IndexError):
            example[4]

        with pytest.raises(IndexError):
            example[-5]

    def test_slice(self):
        selected = example[:]
        assert selected == example

        selected = example[1:-1]
        assert np.allclose(selected.bin_left_edges, [1.4, 1.5])
        assert np.array_equal(selected.frequencies, [0, 3])
        assert np.isclose(selected.underflow, 6)
        assert np.isclose(selected.overflow, 8.2)

    def test_masked(self):
        mask =  np.array([True, True, True, True], dtype=bool)
        assert np.array_equal(example[mask].bins, example.bins)
        assert np.array_equal(example[mask].frequencies, example.frequencies)
        assert np.isnan(example[mask].underflow)
        assert np.isnan(example[mask].overflow)

        mask =  np.array([True, False, False, False], dtype=bool)
        assert np.array_equal(example[mask].bins, example[:1].bins)
        assert np.array_equal(example[mask].frequencies, example[:1].frequencies)
        assert np.isnan(example[mask].underflow)
        assert np.isnan(example[mask].overflow)

        with pytest.raises(IndexError):
            mask =  np.array([True, False, False, False, False, False], dtype=bool)
            example[mask]

        with pytest.raises(IndexError):
            mask =  np.array([False, False, False], dtype=bool)
            example[mask]

    def test_self_condition(self):
        selected = example[example.frequencies > 0]
        assert np.allclose(selected.bin_left_edges, [1.2, 1.5, 1.7])
        assert np.array_equal(selected.frequencies, [4, 3, 7.2])


class TestArithmetic(object):
    def test_add_number(self):
        with pytest.raises(RuntimeError):
            example + 4

    def test_add_wrong_histograms(self):
        with pytest.raises(RuntimeError):
            wrong_bins = [
                 [],                              # No bins
                 [1.2, 1.5, 1.7, 1.8 ],           # Too few
                 [1.2, 1.44, 1.5, 1.7, 1.8],      # Different
                 [1.2, 1.4, 1.5, 1.7, 1.8, 1.]    # Too many
            ]
            values = [1, 1, 0, 2.2, 3, 4, 4]
            for binset in wrong_bins:
                other = Histogram1D(binset, values[:len(binset) - 1])
                with pytest.raises(RuntimeError):
                    example + other

    def test_add_correct_histogram(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [1, 1, 0, 1]
        other = Histogram1D(bins, values)
        sum = example + other
        assert np.allclose(sum.bins, example.bins)
        assert np.allclose(sum.frequencies, [5, 1, 3, 8.2])

    def test_subtract_wrong_histograms(self):
        with pytest.raises(RuntimeError):
            wrong_bins = [
                 [],                              # No bins
                 [1.2, 1.5, 1.7, 1.8 ],           # Too few
                 [1.2, 1.44, 1.5, 1.7, 1.8],      # Different
                 [1.2, 1.4, 1.5, 1.7, 1.8, 1.]    # Too many
            ]
            values = [1, 1, 0, 2.2, 3, 4, 4]
            for binset in wrong_bins:
                other = Histogram1D(binset, values[:len(binset) - 1])
                with pytest.raises(RuntimeError):
                    example - other

    def test_subtract_correct_histogram(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [1, 0, 0, 1]
        other = Histogram1D(bins, values)
        sum = example - other
        assert np.allclose(sum.bins, example.bins)
        assert np.allclose(sum.frequencies, [3, 0, 3, 6.2])

    def test_multiplication(self):
        new = example * 2
        assert new is not example
        assert np.allclose(new.bins, example.bins)
        assert np.allclose(new.frequencies, example.frequencies * 2)
        new *= 2
        assert np.allclose(new.frequencies, example.frequencies * 4)

    def test_rmultiplication(self):
        assert example * 2 == 2 * example

    def test_division(self):
        new = example / 2
        assert new is not example
        assert np.allclose(new.bins, example.bins)
        assert np.allclose(new.frequencies, example.frequencies / 2)
        new /= 2
        assert np.allclose(new.frequencies, example.frequencies / 4)


class TestConversion(object):
    def test_pandas(self):
        df = example.to_dataframe()
        assert df.shape == (4, 3)
        assert np.array_equal(df.columns.values, ["left", "right", "frequency"])
        assert np.array_equal(df.left, [1.2, 1.4, 1.5, 1.7])
        assert np.array_equal(df.right, [1.4, 1.5, 1.7, 1.8 ])
        assert np.array_equal(df.frequency, [4, 0, 3, 7.2])


if __name__ == "__main__":
    pytest.main(__file__)

