import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, note
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes

from physt import h1
from physt.config import config
from physt.histogram1d import Histogram1D
from physt.statistics import INVALID_STATISTICS
from physt.testing.strategies import histograms_1d


@pytest.fixture
def values():
    return np.asarray([4, 0, 3, 7.2])


@pytest.fixture(params=[4, np.asarray([4, 0.17, 3, 7.2], dtype=float), [1, 2, 3, 4]])
def array_like(request):
    return request.param


@pytest.fixture
def bins():
    return np.asarray([1.2, 1.4, 1.5, 1.7, 1.8])


@pytest.fixture
def example_histogram(values, bins):
    return Histogram1D(bins, values, overflow=1, underflow=2)


class TestBins:
    def test_nbins(self, simple_h1: Histogram1D) -> None:
        assert simple_h1.bin_count == 4

    def test_edges(self, simple_h1: Histogram1D) -> None:
        assert np.allclose(simple_h1.bin_left_edges, [0, 1, 1.5, 2])
        assert np.allclose(simple_h1.bin_right_edges, [1, 1.5, 2, 3])
        assert np.allclose(simple_h1.bin_centers, [0.5, 1.25, 1.75, 2.5])

    def test_numpy_bins(self, example_histogram):
        assert np.allclose(example_histogram.numpy_bins, [1.2, 1.4, 1.5, 1.7, 1.8])

    def test_widths(self, example_histogram):
        assert np.allclose(example_histogram.bin_widths, [0.2, 0.1, 0.2, 0.1])


class TestConstructor:
    @pytest.mark.parametrize("free_arithmetics", [True, False])
    def test_negative_values(self, free_arithmetics):
        bins = [[0, 1], [1, 2]]
        values = [-1, 2]
        with config.enable_free_arithmetics(free_arithmetics):
            if free_arithmetics:
                hist = Histogram1D(bins, values)
                assert np.array_equal(hist.frequencies, values)
            else:
                with pytest.raises(ValueError) as ex:
                    _ = Histogram1D(bins, values)
                ex.match("Cannot have negative frequencies.")

    @pytest.mark.parametrize("free_arithmetics", [True, False])
    def test_negative_errors2(self, free_arithmetics):
        bins = [[0, 1], [1, 2]]
        values = [1, 2]
        errors2 = [-1, 1]
        with pytest.raises(ValueError):
            _ = Histogram1D(bins, values, errors2=errors2)


class TestValues:
    def test_values(self, example_histogram):
        assert np.allclose(example_histogram.frequencies, [4, 0, 3, 7.2])

    def test_cumulative_values(self, example_histogram):
        assert np.allclose(example_histogram.cumulative_frequencies, [4, 4, 7, 14.2])

    def test_normalize(self, example_histogram):
        new = example_histogram.normalize()
        expected = np.array([4, 0, 3, 7.2]) / 14.2
        assert np.allclose(new.frequencies, expected)
        assert np.array_equal(new.bins, example_histogram.bins)
        assert new is not example_histogram

        copy = example_histogram.copy()
        new = copy.normalize(inplace=True)
        assert np.allclose(new.frequencies, expected)
        assert np.array_equal(new.bins, example_histogram.bins)
        assert new is copy

    def test_total(self, example_histogram):
        assert np.isclose(example_histogram.total, 14.2)


class TestCopy:
    def test_copy(self, example_histogram):
        new = example_histogram.copy()
        assert new is not example_histogram
        assert new.bins is not example_histogram.bins
        assert new.frequencies is not example_histogram.frequencies
        assert new.statistics is not example_histogram.statistics
        assert new == example_histogram

    def test_copy_no_frequencies(self, example_histogram):
        new = example_histogram.copy(include_frequencies=False)
        assert new is not example_histogram
        assert np.array_equal(new.bins, example_histogram.bins)
        assert new.total == 0
        assert new.overflow == 0
        assert new.underflow == 0

    def test_copy_with_errors(self, bins, values):
        errors2 = [1, 0, 4, 2.6]
        h1 = Histogram1D(bins, values, errors2)
        assert h1.copy() == h1

    def test_copy_meta(self, bins, values):
        errors2 = [1, 0, 4, 2.6]
        h1 = Histogram1D(bins, values, errors2, custom1="custom1", name="name")
        copy = h1.copy()
        assert h1.meta_data == copy.meta_data


class TestEquivalence:
    def test_eq(self, example_histogram, bins, values):
        other1 = Histogram1D(bins, values, underflow=2, overflow=1)
        assert other1 == example_histogram

        bins = [1.22, 1.4, 1.5, 1.7, 1.8]
        values = [4, 0, 3, 7.2]
        other2 = Histogram1D(bins, values, underflow=2, overflow=1)
        assert other2 != example_histogram

        bins = [1.2, 1.4, 1.5, 1.7, 1.8]
        values = [4, 0, 13, 7.2]
        other3 = Histogram1D(bins, values, underflow=2, overflow=1)
        assert other3 != example_histogram

        bins = [1.2, 1.4, 1.5, 1.7, 1.8]
        values = [4, 0, 3, 7.2]
        errors2 = [4, 0, 3, 7.2]
        other4 = Histogram1D(bins, values, errors2, underflow=2, overflow=1)
        assert other4 == example_histogram

        errors2 = [4, 0, 3, 8.2]
        other5 = Histogram1D(bins, values, errors2, underflow=2, overflow=1)
        assert other5 != example_histogram

    def test_eq_with_underflows(self, example_histogram):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8]
        values = [4, 0, 3, 7.2]
        other1 = Histogram1D(bins, values, underflow=2)
        assert other1 != example_histogram

        other2 = Histogram1D(bins, values, overflow=1)
        assert other2 != example_histogram

        other3 = Histogram1D(bins, values, overflow=1, underflow=2)
        assert other3 == example_histogram


class TestIndexing:
    def test_single(self, example_histogram):
        zeroth = example_histogram[0]
        assert np.allclose(zeroth[0], (1.2, 1.4))
        assert zeroth[1] == 4

        other = example_histogram[-2]
        assert np.allclose(other[0], (1.5, 1.7))
        assert other[1] == 3

        with pytest.raises(IndexError):
            example_histogram[4]

        with pytest.raises(IndexError):
            example_histogram[-5]

    def test_slice(self, example_histogram):
        selected = example_histogram[:]
        assert selected == example_histogram

        selected = example_histogram[1:-1]
        assert np.allclose(selected.bin_left_edges, [1.4, 1.5])
        assert np.array_equal(selected.frequencies, [0, 3])
        assert np.isclose(selected.underflow, 6)
        assert np.isclose(selected.overflow, 8.2)

    def test_slice_with_upper_bound(self, example_histogram):
        selected = example_histogram[:3]
        assert np.array_equal(selected.frequencies, [4, 0, 3])

    def test_masked(self, example_histogram):
        mask = np.array([True, True, True, True], dtype=bool)
        assert np.array_equal(example_histogram[mask].bins, example_histogram.bins)
        assert np.array_equal(
            example_histogram[mask].frequencies, example_histogram.frequencies
        )
        assert np.isnan(example_histogram[mask].underflow)
        assert np.isnan(example_histogram[mask].overflow)

        mask = np.array([True, False, False, False], dtype=bool)
        assert np.array_equal(example_histogram[mask].bins, example_histogram[:1].bins)
        assert np.array_equal(
            example_histogram[mask].frequencies, example_histogram[:1].frequencies
        )
        assert np.isnan(example_histogram[mask].underflow)
        assert np.isnan(example_histogram[mask].overflow)

        with pytest.raises(IndexError):
            mask = np.array([True, False, False, False, False, False], dtype=bool)
            example_histogram[mask]

        with pytest.raises(IndexError):
            mask = np.array([False, False, False], dtype=bool)
            example_histogram[mask]

    def test_array(self, example_histogram):
        selected = example_histogram[[1, 2]]
        assert np.allclose(selected.bin_left_edges, [1.4, 1.5])
        assert np.array_equal(selected.frequencies, [0, 3])
        assert np.isnan(selected.underflow)
        assert np.isnan(selected.overflow)

    def test_self_condition(self, example_histogram):
        selected = example_histogram[example_histogram.frequencies > 0]
        assert np.allclose(selected.bin_left_edges, [1.2, 1.5, 1.7])
        assert np.array_equal(selected.frequencies, [4, 3, 7.2])


class TestArithmetics:
    class TestAddition:
        @pytest.mark.parametrize("free_arithmetics", [True, False])
        def test_add_non_hist(
            self, free_arithmetics, example_histogram, values, array_like
        ):
            with config.enable_free_arithmetics(free_arithmetics):
                if free_arithmetics:
                    result = example_histogram + array_like
                    assert np.array_equal(result.frequencies, values + array_like)
                    assert np.array_equal(
                        result.errors2, example_histogram.errors2 + array_like
                    )
                    assert np.isnan(result.missed)
                    assert result.statistics == INVALID_STATISTICS

                    example_histogram += array_like
                    assert example_histogram == result
                else:
                    with pytest.raises(
                        TypeError, match="Only histograms can be added together."
                    ):
                        example_histogram + array_like

        def test_add_wrong_histograms(self, example_histogram):
            with pytest.raises(ValueError):
                wrong_bins = [
                    [],  # No bins
                    [1.2, 1.5, 1.7, 1.8],  # Too few
                    [1.2, 1.44, 1.5, 1.7, 1.8],  # Different
                    [1.2, 1.4, 1.5, 1.7, 1.8, 1.0],  # Too many
                ]
                values = [1, 1, 0, 2.2, 3, 4, 4]
                for binset in wrong_bins:
                    other = Histogram1D(binset, values[: len(binset) - 1])
                    with pytest.raises(ValueError):
                        example_histogram + other

        def test_add_correct_histogram(self, bins, example_histogram):
            other_values = [1, 1, 0, 1]
            other = Histogram1D(bins, other_values)
            sum = example_histogram + other
            assert np.allclose(sum.bins, example_histogram.bins)
            assert np.allclose(sum.frequencies, [5, 1, 3, 8.2])

        def test_adding_with_meta_data(self, example_histogram):
            e1 = example_histogram.copy()
            e2 = example_histogram.copy()
            e3 = example_histogram.copy()
            e4 = example_histogram.copy()
            e1.name = "a"
            e2.name = "b"
            e3.name = "a"
            e4.name = None
            assert (e1 + e2).name is None
            assert (e1 + e3).name == "a"
            assert (e1 + e4).name is None

    class TestSubtraction:
        def test_subtract_wrong_histograms(self, example_histogram):
            wrong_bins = [
                [1.2, 1.5, 1.7, 1.8],  # Too few
                [1.2, 1.44, 1.5, 1.7, 1.8],  # Different
                [1.2, 1.4, 1.5, 1.7, 1.8, 2.0],  # Too many
            ]
            values = [1, 1, 0, 2.2, 3, 4, 4]
            for binset in wrong_bins:
                other = Histogram1D(binset, values[: len(binset) - 1])
                with pytest.raises(ValueError):
                    example_histogram - other

        def test_subtract_correct_histogram(self, example_histogram, bins):
            other_values = [1, 0, 0, 1]
            other = Histogram1D(bins, other_values)
            sum = example_histogram - other
            assert np.allclose(sum.bins, example_histogram.bins)
            assert np.allclose(sum.frequencies, [3, 0, 3, 6.2])

        @pytest.mark.parametrize("free_arithmetics", [True, False])
        def test_subtract_non_hist(
            self, free_arithmetics, example_histogram, values, array_like
        ):
            with config.enable_free_arithmetics(free_arithmetics):
                if free_arithmetics:
                    result = example_histogram - array_like
                    assert np.array_equal(result.frequencies, values - array_like)
                    assert np.array_equal(
                        result.errors2, example_histogram.errors2 + array_like
                    )
                    assert np.isnan(result.missed)
                    assert result.statistics == INVALID_STATISTICS

                    example_histogram -= array_like
                    assert example_histogram == result
                else:
                    with pytest.raises(
                        TypeError, match="Only histograms can be added together."
                    ):
                        example_histogram + array_like

    class TestMultiplication:
        @pytest.mark.parametrize("free_arithmetics", [True, False])
        def test_multiplication_non_hist(
            self, example_histogram, free_arithmetics, values, array_like
        ):
            with config.enable_free_arithmetics(free_arithmetics):
                if free_arithmetics or np.isscalar(array_like):
                    new = example_histogram * array_like
                    assert new is not example_histogram
                    assert np.allclose(new.bins, example_histogram.bins)
                    assert np.allclose(new.frequencies, values * array_like)

                    example_histogram *= array_like
                    assert example_histogram == new
                else:
                    with pytest.raises(TypeError):
                        example_histogram * array_like

        def test_multiplication_hist(self, example_histogram):
            with pytest.raises(
                TypeError, match="^Multiplication of two histograms is not supported.$"
            ):
                example_histogram * example_histogram

        def test_rmultiplication_scalar(self, example_histogram):
            assert example_histogram * 2 == 2 * example_histogram

    class TestDivision:
        @pytest.mark.parametrize("free_arithmetics", [True, False])
        def test_division_non_hist(
            self, example_histogram, free_arithmetics, values, array_like
        ):
            with config.enable_free_arithmetics(free_arithmetics):
                if free_arithmetics or np.isscalar(array_like):
                    new = example_histogram / array_like
                    assert new is not example_histogram
                    assert np.allclose(new.bins, example_histogram.bins)
                    assert np.allclose(new.frequencies, values / array_like)

                    example_histogram /= array_like
                    assert example_histogram == new
                else:
                    with pytest.raises(TypeError):
                        example_histogram * array_like

        def test_division_two_hist(self, example_histogram):
            with pytest.raises(
                TypeError, match="^Division of two histograms is not supported.$"
            ):
                example_histogram / example_histogram

        def test_division_scalar_hist(self, example_histogram):
            with pytest.raises(TypeError, match="unsupported operand type"):
                1 / example_histogram


class TestMerging:
    def test_2(self):
        data = np.random.rand(100)
        hist1 = h1(data, 120)
        hist2 = h1(data, 60)
        merged = hist1.merge_bins(2, inplace=False)
        assert hist2 == merged


class TestFindBin:
    def test_normal(self, example_histogram):
        # bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        assert example_histogram.find_bin(1) == -1
        assert example_histogram.find_bin(1.3) == 0
        assert example_histogram.find_bin(1.5) == 2
        assert example_histogram.find_bin(1.72) == 3
        assert example_histogram.find_bin(1.9) == 4
        assert example_histogram.find_bin(1.8) == 3

    def test_inconsecutive(self, example_histogram):
        selected = example_histogram[[0, 3]]
        assert selected.find_bin(1) == -1
        assert selected.find_bin(1.3) == 0
        assert selected.find_bin(1.45) is None
        assert selected.find_bin(1.55) is None
        assert selected.find_bin(1.75) == 1
        assert selected.find_bin(1.8) == 1
        assert selected.find_bin(1.9) == 2


class TestFill:
    @pytest.mark.skip(reason="hypothesis slow")
    @given(histogram=histograms_1d(), value=st.floats(), weight=st.floats(min_value=0))
    def test_increases_total_by_zero_or_weight(
        self, value: float, histogram: Histogram1D, weight
    ):
        # We let the OverflowError to occur
        assume(weight < 1e100)
        note(f"Pre: {histogram!r}")
        pre = histogram.total
        histogram.fill(value, weight=weight)
        note(f"Post: {histogram!r}")
        post = histogram.total
        assert (post == pre) or (np.isclose(post, pre + weight, rtol=1e-8))

    def test_fill(self, example_histogram):
        # bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        # values = [4, 0, 3, 7.2]
        copy = example_histogram.copy()
        copy.fill(1.44)  # Underflow
        assert np.allclose(copy.frequencies, [4, 1, 3, 7.2])

        copy.fill(1.94)  # Overflow
        assert np.allclose(copy.frequencies, [4, 1, 3, 7.2])
        assert copy.overflow == 2

        copy.fill(0.44)  # Underflow
        assert np.allclose(copy.frequencies, [4, 1, 3, 7.2])
        assert copy.underflow == 3

        copy.fill(1.44, weight=2.2)
        assert np.allclose(copy.frequencies, [4, 3.2, 3, 7.2])

    def test_fill_dtype(self):
        h = Histogram1D([[0, 1], [1, 2], [2, 3]], [1, 2, 3])
        assert h.dtype == np.int_
        assert np.allclose(h.frequencies, [1, 2, 3])

        h.fill(1.3, weight=2.2)
        # assert h.dtype == np.float
        assert np.allclose(h.frequencies, [1, 4.2, 3])


class TestFillN:
    @given(
        histogram=histograms_1d(),
        values=arrays(
            dtype=floating_dtypes(), shape=array_shapes(min_dims=1, max_dims=1)
        ),
    )
    def test_increases_total_by_zero_or_weight(self, histogram, values):
        pre = histogram.total
        histogram.fill_n(values)
        post = histogram.total
        assert pre <= post <= np.nextafter(pre + values.size, np.inf)

    def test_mismatch_shape(self):
        pass


class TestDtype:
    def test_simple(self, values):
        hist = h1(values)
        assert hist.dtype == np.int_

    def test_with_weights(self, values):
        hist = h1(values, weights=[1, 2, 2.1, 3.2])
        assert hist.dtype == float

    def test_float_weights_in_integer(self, values):
        with pytest.raises(ValueError):
            h1(values, weights=[1, 2, 2.1, 3.2], dtype=int)

    @pytest.mark.parametrize("dtype", Histogram1D.SUPPORTED_DTYPES)
    @pytest.mark.parametrize(
        "values",
        [pytest.param([0, 1, 1], id="existing"), pytest.param(None, id="none")],
    )
    def test_explicit_construction(self, dtype, values):
        hist = h1(values, dtype=dtype, range=(-1, 1))
        assert hist.dtype == dtype

    def test_copy(self, values):
        hist = h1(values, dtype=np.int32)
        assert hist.dtype == np.int32
        assert hist.copy().dtype == np.int32

    def test_coerce(self, values):
        hist = h1(values, dtype=np.int32)
        hist._coerce_dtype(np.int64)
        assert hist.dtype == np.int64
        hist._coerce_dtype(float)
        assert hist.dtype == float
        hist._coerce_dtype(np.int32)
        assert hist.dtype == float

    def test_update(self, values):
        hist = h1(values)
        hist.dtype = np.int16
        assert hist.dtype == np.int16
        assert hist.frequencies.dtype == np.int16

        hist = h1(values, weights=[1, 2, 2.1, 3.2])
        with pytest.raises(ValueError):
            hist.dtype = np.int16

        hist = h1(values, weights=[1, 2, 2, 3])
        hist.dtype = np.int16
        assert hist.dtype == np.int16

    def test_hist_arithmetic(self, values):
        hist1 = h1(values, dtype=np.int32)
        hist2 = hist1.copy()
        hist2.dtype = float
        hist2 *= 1.01

        example3 = hist1.copy()
        example3.dtype = np.int64

        assert (hist1 + hist2).dtype == float
        assert (hist2 + hist1).dtype == float
        assert (hist1 + example3).dtype == np.int64
        assert (example3 - hist1).dtype == np.int64

        hist1 += hist2
        assert hist1.dtype == float

    def test_scalar_arithmetic(self, values):
        hist = h1(values, dtype=np.int32)

        assert (hist / 3).dtype == float
        assert (hist * 3).dtype in (np.int32, np.int64)  # Different platforms :-/
        assert (hist * 3.1).dtype == float

        with pytest.raises(TypeError):
            hist * complex(4, 5)

    def test_empty(self):
        hist = h1(None, "fixed_width", bin_width=10, adaptive=True)
        assert hist.dtype == np.int64
