import numpy as np

from physt import h1


class TestNumpyBins:
    def test_nbin(self):
        arr = np.random.rand(100)
        hist = h1(arr, bins=15)
        assert hist.bin_count == 15
        assert np.isclose(hist.bin_right_edges[-1], arr.max())
        assert np.isclose(hist.bin_left_edges[0], arr.min())

    def test_edges(self):
        arr = np.arange(0, 1, 0.01)
        hist = h1(arr, np.arange(0.1, 0.8001, 0.1))
        assert np.allclose(hist.numpy_bins, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        assert hist.underflow == 10
        assert hist.overflow == 19

    def test_range(self):
        arr = np.arange(0, 1.00, 0.01)
        hist = h1(arr, 10, range=(0.5, 1.0))
        assert hist.bin_count == 10
        assert hist.bin_left_edges[0] == 0.5
        assert hist.bin_right_edges[-1] == 1.0
        assert hist.overflow == 0
        assert hist.underflow == 50
        assert hist.total == 50

        hist = h1(arr, bins=10, range=(0.5, 1.0), keep_missed=False)
        assert hist.total == 50
        assert np.isnan(hist.underflow)
        assert np.isnan(hist.overflow)

    def test_metadata(self):
        arr = np.arange(0, 1.00, 0.01)
        hist = h1(arr, name="name", title="title", axis_name="axis_name")
        assert hist.name == "name"
        assert hist.title == "title"
        assert hist.axis_names == ("axis_name",)
