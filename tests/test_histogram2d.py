import numpy as np
import pytest

import physt
from physt import h2, histogram_nd, binnings
from physt.binnings import as_binning, BinningBase
from physt.histogram_nd import Histogram2D

vals = [
    [0.1, 2.0],
    [-0.1, 0.7],
    [0.2, 1.5],
    [0.2, -1.5],
    [0.2, 1.47],
    [1.2, 1.23],
    [0.7, 0.5]
]

np.random.seed(42)


@pytest.fixture
def bins0to3() -> BinningBase:
    return as_binning(np.linspace(0, 3, 4))


@pytest.fixture
def a3x3() -> np.ndarray:
    return np.linspace(0, 8, 9).reshape((3, 3))


@pytest.fixture
def h3x3(bins0to3, a3x3) -> Histogram2D:
    """Simple 2D histograms of shape 3x3."""
    return Histogram2D(
        binnings=[bins0to3, bins0to3],
        frequencies=a3x3
    )


class TestCalculateFrequencies:
    def test_simple(self):
        bins = [[0, 1, 2], [0, 1, 2]]
        schemas = [binnings.static_binning(None, np.asarray(bs)) for bs in bins]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, binnings=schemas)
        assert np.array_equal([[1, 3], [0, 1]], frequencies)
        assert missing == 2
        assert errors2 is None

    def test_gap(self):
        bins = [
            [[-1, 0], [1, 2]],
            [[-2, -1], [1, 2]]
        ]
        schemas = [binnings.static_binning(None, np.asarray(bs)) for bs in bins]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, binnings=schemas)
        assert np.array_equal([[0, 0], [0, 1]], frequencies)
        assert missing == 6
        assert errors2 is None

    def test_errors(self):
        bins = [
            [[-1, 0], [1, 2]],
            [[-2, -1], [1, 2]]
        ]
        weights = [2, 1, 1, 1, 1, 2, 1]
        schemas = [binnings.static_binning(None, np.asarray(bs)) for bs in bins]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, binnings=schemas, weights=weights)
        assert np.array_equal([[0, 0], [0, 2]], frequencies)
        assert missing == 7
        assert np.array_equal([[0, 0], [0, 4]], errors2)


class TestHistogram2D:
    def test_simple_random(self):
        x = np.random.normal(100, 1, 1000)
        y = np.random.normal(10, 10, 1000)
        h2 = physt.h2(x, y, [8, 4], name="Some histogram", axis_names=["x", "y"])
        assert h2.frequencies.sum() == 1000
        assert h2.shape == (8, 4)
        assert h2.name == "Some histogram"
        assert h2.axis_names == ("x", "y")

    def test_dropna(self):
        vals2 = np.array(vals)
        vals2[0, 1] = np.nan
        with pytest.raises(RuntimeError):
            hist = physt.h2(vals2[:,0], vals2[:,1], dropna=False)
        hist = physt.h2(vals2[:, 0], vals2[:, 1])
        assert hist.frequencies.sum() == 6


class TestArithmetics:
    # TODO: Add free arithmetics checks
    def test_multiply_by_constant(self, h3x3, a3x3):
        mul1 = h3x3 * 2
        assert np.array_equal(mul1.frequencies, a3x3 * 2)
        assert np.array_equal(mul1.errors2, a3x3 * 4)

        mul2 = h3x3 * 0.5
        assert np.array_equal(mul2.frequencies, a3x3 * 0.5)
        assert np.array_equal(mul2.errors2, a3x3 * 0.25)

    def test_multiply_by_other(self, h3x3):
        with pytest.raises(TypeError):
            h3x3 * h3x3

    def test_divide_by_other(self, h3x3):
        with pytest.raises(TypeError):
            h3x3 / h3x3

    def test_divide_by_constant(self, h3x3, a3x3):
        frac = h3x3 / 2
        assert np.array_equal(frac.frequencies, a3x3 / 2)
        assert np.array_equal(frac.errors2, a3x3 / 4)

    def test_addition_by_constant(self, h3x3):
        with pytest.raises(TypeError):
            h3x3 + 4

    def test_addition_with_another(self, h3x3, a3x3):
        add = h3x3 + h3x3
        assert np.array_equal(add.frequencies, a3x3 * 2)
        assert np.array_equal(add.errors2, a3x3 * 2)

    def test_addition_with_adaptive(self, create_adaptive):
        ha = create_adaptive((1, 2))
        hb = create_adaptive((2, 2))
        hha = ha + hb
        assert hha == hb + ha
        assert hha.shape == (2, 2)
        assert hha.total == 7
        assert np.array_equal(hha.frequencies, [[0, 2], [2, 3]])

    def test_subtraction_with_another(self, h3x3, a3x3):
        sub = h3x3 * 2 - h3x3
        assert np.array_equal(sub.frequencies, a3x3)
        assert np.array_equal(sub.errors2, 5 * a3x3)

    def test_subtraction_by_constant(self, h3x3):
        with pytest.raises(TypeError):
            h3x3 - 4


class TestDtype:
    def test_simple(self):
        from physt import examples
        assert examples.normal_h2().dtype == np.dtype(np.int64)


class TestMerging:
    def test_2(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        hh = h2(data1, data2, 120)
        hha = h2(data1, data2, 60)
        hhb = hh.merge_bins(2, inplace=False)
        assert hha == hhb


class TestPartialNormalizing:
    def test_wrong_arguments(self):
        freqs = [
            [1, 0],
            [1, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs)
        with pytest.raises(ValueError):
            h0 = h.partial_normalize(2)
        with pytest.raises(ValueError):
            h0 = h.partial_normalize(-2)

    def test_axis_names(self):
        freqs = [
            [1, 0],
            [1, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs, axis_names=["first_axis", "second_axis"])
        h1 = h.partial_normalize("second_axis")
        assert np.allclose(h1.frequencies, [[1, 0], [.333333333333, .6666666666]])
        with pytest.raises(ValueError):
            h0 = h.partial_normalize("third_axis")

    def test_inplace(self):
        freqs = [
            [1, 0],
            [1, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs)
        h1 = h.partial_normalize(0, inplace=False)
        assert np.allclose(h.frequencies, freqs)
        assert not np.allclose(h1.frequencies, h.frequencies)
        h.partial_normalize(0, inplace=True)
        assert h1 == h

    def test_values(self):
        freqs = [
            [1, 0],
            [1, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs)
        h0 = h.partial_normalize(0)
        h1 = h.partial_normalize(1)

        assert np.allclose(h0.frequencies, [[.5, 0], [.5, 1.0]])
        assert np.allclose(h1.frequencies, [[1, 0], [.333333333333, .6666666666]])

    def test_with_zeros(self):
        freqs = [
            [0, 0],
            [0, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs)
        h1 = h.partial_normalize(1)
        assert np.allclose(h1.frequencies, [[0, 0], [0, 1.0]])
