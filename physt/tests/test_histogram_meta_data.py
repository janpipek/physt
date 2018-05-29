import unittest

from physt.histogram_meta_data import HistogramMetaData


class TestHistogramMetaData(unittest.TestCase):
    def test_init(self):
        meta_data = HistogramMetaData()
        assert len(meta_data.keys()) == 0

    def test_title(self):
        meta_data = HistogramMetaData()
        assert meta_data.title is None

        meta_data = HistogramMetaData({"title": "aaa"})
        assert meta_data.title == "aaa"

        meta_data.title = "bbb"
        assert meta_data.title == "bbb"
        assert meta_data["title"] == "bbb"