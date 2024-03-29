import numpy as np
import pytest

from physt import h2, h3
from physt._facade import histogramdd
from physt.types import Histogram1D, Histogram2D, HistogramND
from physt.typing_aliases import ArrayLike


class TestHistogramND:
    def test_creation(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        data3 = np.random.rand(100)
        data = np.array([data1, data2, data3]).T
        h = histogramdd(data)
        assert h.ndim == 3

    def test_copy(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        data3 = np.random.rand(100)
        data = np.array([data1, data2, data3]).T
        h = histogramdd(data)
        print(h._binnings)
        h2 = h.copy()
        assert h == h2
        assert np.array_equal(h.bins, h2.bins)

    def test_total_size(self):
        data = np.random.rand(100)
        h = h2(data, data, range=(0, 0.5))
        assert h.total_size == 0.25

    def test_bin_sizes(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        data3 = np.random.rand(100)
        data = np.array([data1, data2, data3]).T
        h = histogramdd(data, [10, 11, 12])
        assert h.bin_sizes.shape == (10, 11, 12)

    class TestFindBin:
        def test_scalar_with_axis(self, simple_h2: Histogram2D) -> None:
            assert 1 == simple_h2.find_bin(1.7, axis="x")

        def test_scalar_without_axis(self, simple_h2: Histogram2D) -> None:
            with pytest.raises(TypeError, match="Array expected"):
                simple_h2.find_bin(5)

        @pytest.mark.parametrize("value", [-1, 17, np.inf])
        def test_scalar_outside_the_range(
            self, simple_h2: Histogram2D, value: float
        ) -> None:
            assert simple_h2.find_bin(value, axis="x") is None

        @pytest.mark.parametrize("value", [[0.5, 6.5], [3.5, 5.5], [-np.inf, 12]])
        def test_array_outside_the_range(
            self, simple_h2: Histogram2D, value: ArrayLike
        ) -> None:
            assert simple_h2.find_bin(value) is None

        def test_array_with_axis(self, simple_h2: Histogram2D) -> None:
            with pytest.raises(TypeError, match="Number expected"):
                simple_h2.find_bin([2.7, 4.4], axis="x")

        def test_array_without_axis(self, simple_h2: Histogram2D) -> None:
            assert (2, 0) == simple_h2.find_bin([2.7, 4.4])

        def test_array_with_wrong_shape(self, simple_h2: Histogram2D) -> None:
            with pytest.raises(ValueError, match="Wrong shape"):
                simple_h2.find_bin([2.2, 2.2, 2.2])


class TestProjections:
    def test_4_to_3(self):
        data = np.random.rand(100, 4)
        h = histogramdd(data, [4, 5, 6, 7], axis_names=["1", "2", "3", "4"])
        h3 = h.projection(1, 2, 3)
        assert h3.ndim == 3
        assert isinstance(h3, HistogramND)
        assert h3.total == h.total
        assert h3.axis_names == ("2", "3", "4")
        assert h3.frequencies.shape == (5, 6, 7)
        assert h3.shape == (5, 6, 7)

    def test_4_to_2(self):
        data = np.random.rand(100, 4)
        h = histogramdd(data, [4, 5, 6, 7], axis_names=["1", "2", "3", "4"])
        h2 = h.projection(1, 3)
        assert isinstance(h2, Histogram2D)
        assert h2.total == h.total
        assert h2.axis_names == ("2", "4")
        assert h2.shape == h2.frequencies.shape
        assert h2.shape == (5, 7)

    def test_3_to_2(self):
        data = np.random.rand(100, 3)
        h = histogramdd(data, [4, 5, 6], axis_names=["1", "2", "3"])
        h2 = h.projection(1, 2)
        assert isinstance(h2, Histogram2D)
        assert h2.total == h.total
        assert h2.axis_names == ("2", "3")
        assert h2.shape == h2.frequencies.shape
        assert h2.shape == (5, 6)

    def test_2_to_1(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        h = h2(data1, data2, [4, 5], axis_names=["1", "2"])
        projection = h.projection(1)
        assert isinstance(projection, Histogram1D)
        assert projection.total == h.total
        assert projection.axis_name == "2"
        assert projection.shape == projection.frequencies.shape
        assert projection.shape == (5,)

    def test_projection_by_name(self):
        data = np.random.rand(100, 4)
        h = histogramdd(data, [4, 5, 6, 7], axis_names=["1", "2", "3", "4"])
        h3 = h.projection("2", "3", "4")
        assert h3.ndim == 3
        assert isinstance(h3, HistogramND)
        assert h3.total == h.total
        assert h3.axis_names == ("2", "3", "4")
        assert h3.frequencies.shape == (5, 6, 7)
        assert h3.shape == (5, 6, 7)

    def test_invalid(self):
        data = np.random.rand(100, 4)
        h = histogramdd(data, [4, 5, 6, 7], axis_names=["1", "2", "3", "4"])
        with pytest.raises(ValueError):
            h.projection("1", "1")
        with pytest.raises(ValueError):
            h.projection("0", "1")
        with pytest.raises(ValueError):
            h.projection("1", "2", "3", "4", "5")
        with pytest.raises(ValueError):
            h.projection()


class TestSlicing:
    def test_slicing_with_upper_bound_only(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        h = h2(data1, data2, [4, 5], axis_names=["1", "2"])
        assert h[:2].shape == (2, 5)

    def test_shapes(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        h = h2(data1, data2, [4, 5], axis_names=["1", "2"])
        assert h[2].shape == (5,)
        assert h[:, 2].shape == (4,)
        # TODO: Add more combinations


class TestH2:
    def test_create_empty_h2(self):
        h2(None, None, "integer", adaptive=True)


class TestH3:
    def test_create_empty_h3(self):
        h3(None, "integer", adaptive=True)
