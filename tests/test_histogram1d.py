from __future__ import division
import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
from physt.histogram1d import Histogram1D
from physt import h1
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
        expected = np.array([4, 0, 3, 7.2]) / 14.2
        assert np.allclose(new.frequencies, expected)
        assert np.array_equal(new.bins, example.bins)
        assert new is not example

        copy = example.copy()
        new = copy.normalize(inplace=True)
        assert np.allclose(new.frequencies, expected)
        assert np.array_equal(new.bins, example.bins)
        assert new is copy

    def test_total(self):
        assert np.isclose(example.total, 14.2)


class TestCopy(object):
    def test_copy(self):
        new = example.copy()
        assert new is not example
        assert new.bins is not example.bins
        assert new.frequencies is not example.frequencies
        assert new == example

    def test_copy_no_frequencies(self):
        new = example.copy(include_frequencies=False)
        assert new is not example
        assert np.array_equal(new.bins, example.bins)
        assert new.total == 0
        assert new.overflow == 0
        assert new.underflow ==0

    def test_copy_with_errors(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [4, 0, 3, 7.2]
        errors2 = [1, 0, 4, 2.6]
        h1 = Histogram1D(bins, values, errors2)
        assert h1.copy() == h1

    def test_copy_meta(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [4, 0, 3, 7.2]
        errors2 = [1, 0, 4, 2.6]
        h1 = Histogram1D(bins, values, errors2, custom1="custom1", name="name")
        copy = h1.copy()
        assert h1.meta_data == copy.meta_data


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

        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [4, 0, 3, 7.2]
        errors2 = [4, 0, 3, 7.2]
        other4 = Histogram1D(bins, values, errors2, underflow=2, overflow=1)
        assert other4 == example

        errors2 = [4, 0, 3, 8.2]
        other5 = Histogram1D(bins, values, errors2, underflow=2, overflow=1)
        assert other5 != example

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

    def test_slice_with_upper_bound(self):
        selected = example[:3]
        assert np.array_equal(selected.frequencies, [4, 0, 3])

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

    def test_array(self):
        selected = example[[1, 2]]
        assert np.allclose(selected.bin_left_edges, [1.4, 1.5])
        assert np.array_equal(selected.frequencies, [0, 3])
        assert np.isnan(selected.underflow)
        assert np.isnan(selected.overflow)

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

    def test_adding_with_meta_data(self):
        e1 = example.copy()
        e2 = example.copy()
        e3 = example.copy()
        e4 = example.copy()
        e1.name = "a"
        e2.name = "b"
        e3.name = "a"
        e4.name = None
        assert (e1 + e2).name == None
        assert (e1 + e3).name == "a"
        assert (e1 + e4).name == None

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


class TestMerging(object):
    def test_2(self):
        data = np.random.rand(100)
        hh = h1(data, 120)
        hha = h1(data, 60)
        hhb = hh.merge_bins(2, inplace=False)
        assert hha == hhb


class TestConversion(object):
    def test_pandas(self):
        df = example.to_dataframe()
        assert df.shape == (4, 4)
        assert np.array_equal(df.columns.values, ["left", "right", "frequency", "error"])
        assert np.array_equal(df.left, [1.2, 1.4, 1.5, 1.7])
        assert np.array_equal(df.right, [1.4, 1.5, 1.7, 1.8 ])
        assert np.array_equal(df.frequency, [4, 0, 3, 7.2])

    # def test_json(self):
    #     json = example.to_json()
    #     h2 = Histogram1D.from_json(json)
    #     assert example == h2


class TestFindBin(object):
    def test_normal(self):
        # bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        assert example.find_bin(1) == -1
        assert example.find_bin(1.3) == 0
        assert example.find_bin(1.5) == 2
        assert example.find_bin(1.72) == 3
        assert example.find_bin(1.9) == 4
        assert example.find_bin(1.8) == 3

    def test_inconsecutive(self):
        selected = example[[0, 3]]
        assert selected.find_bin(1) == -1
        assert selected.find_bin(1.3) == 0
        assert selected.find_bin(1.45) is None
        assert selected.find_bin(1.55) is None
        assert selected.find_bin(1.75) == 1
        assert selected.find_bin(1.8) == 1
        assert selected.find_bin(1.9) == 2


class TestFill(object):
    def test_fill(self):
        # bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        # values = [4, 0, 3, 7.2]
        copy = example.copy()
        copy.fill(1.44)    # Underflow
        assert np.allclose(copy.frequencies, [4, 1, 3, 7.2])

        copy.fill(1.94)    # Overflow
        assert np.allclose(copy.frequencies, [4, 1, 3, 7.2])
        assert copy.overflow == 2

        copy.fill(0.44)    # Underflow
        assert np.allclose(copy.frequencies, [4, 1, 3, 7.2])
        assert copy.underflow == 3

        copy.fill(1.44, weight=2.2)
        assert np.allclose(copy.frequencies, [4, 3.2, 3, 7.2])

    def test_fill_dtype(self):
        h = Histogram1D([[0,1], [1, 2], [2, 3]], [1, 2, 3])
        assert h.dtype == np.int64
        assert np.allclose(h.frequencies, [1, 2, 3])

        h.fill(1.3, weight=2.2)
        # assert h.dtype == np.float
        assert np.allclose(h.frequencies, [1, 4.2, 3])


class TestDtype(object):
    def test_simple(self):
        example = h1(values)
        assert example.dtype == np.int64

    def test_with_weights(self):
        example = h1(values, weights=[1, 2, 2.1, 3.2])
        assert example.dtype == np.float

    def test_explicit(self):
        example = h1(values, dtype=float)
        assert example.dtype == float

        with pytest.raises(RuntimeError):
            example = h1(values, weights=[1, 2, 2.1, 3.2], dtype=int)

    def test_copy(self):
        example = h1(values, dtype=np.int32)
        assert example.dtype == np.int32
        assert example.copy().dtype == np.int32

    def test_coerce(self):
        example = h1(values, dtype=np.int32)
        example._coerce_dtype(np.int64)
        assert example.dtype == np.int64
        example._coerce_dtype(np.float)
        assert example.dtype == np.float
        example._coerce_dtype(np.int32)
        assert example.dtype == np.float

    def test_update(self):
        example = h1(values)
        example.dtype = np.int16
        assert example.dtype == np.int16
        assert example.frequencies.dtype == np.int16

        example = h1(values, weights=[1, 2, 2.1, 3.2])
        with pytest.raises(RuntimeError):
            example.dtype = np.int16

        example = h1(values, weights=[1, 2, 2, 3])
        example.dtype = np.int16
        assert example.dtype == np.int16

    def test_hist_arithmetic(self):
        example = h1(values, dtype=np.int32)
        example2 = example.copy()
        example2.dtype = np.float
        example2 *= 1.01

        example3 = example.copy()
        example3.dtype = np.int64

        assert (example + example2).dtype == np.float
        assert (example2 + example).dtype == np.float
        assert (example + example3).dtype == np.int64
        assert (example3 - example).dtype == np.int64

        example += example2
        assert example.dtype == np.float

    def test_scalar_arithmetic(self):
        example = h1(values, dtype=np.int32)

        assert (example / 3).dtype == np.float
        assert (example * 3).dtype == np.int32
        assert (example * 3.1).dtype == np.float

        with pytest.raises(TypeError):
            example * complex(4, 5)


if __name__ == "__main__":
    pytest.main(__file__)
