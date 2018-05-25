import unittest
from unittest.mock import Mock, patch

import numpy as np

from physt.schema import (build_schema, StaticSchema, NumpySchema, HumanSchema,
                          UnknownBinCountAlgorithmError)


class TestBuildSchema(unittest.TestCase):
    def test_no_params(self):
        schema = build_schema()
        assert isinstance(schema, HumanSchema)

    def test_numpy(self):
        schema = build_schema("numpy")
        assert isinstance(schema, NumpySchema)
        assert schema.bin_arg == 10
        assert schema.edges is None

        schema = build_schema("numpy", bins=12)
        assert isinstance(schema, NumpySchema)
        assert schema.bin_arg == 12
        assert schema.edges is None

        # TODO: Test fail with invalid data type for bins

    def test_unknown_schema(self):
        from physt.schema import UnknownSchemaError
        with self.assertRaises(UnknownSchemaError):
            _ = build_schema("invalid")

    def test_integer(self):
        schema = build_schema(bins=4)
        assert isinstance(schema, NumpySchema)
        assert schema.bin_arg == 4

    def test_string(self):
        algo = "sqrt"
        schema = build_schema(bins=algo)
        assert isinstance(schema, NumpySchema)
        assert schema.bin_arg == algo

        algo = "nonexistent"
        with self.assertRaises(UnknownBinCountAlgorithmError):
            schema = build_schema(bins=algo)

    def test_edge_array(self):
        bins = [0, 1, 2, 3]
        schema = build_schema(bins=bins)
        assert isinstance(schema, StaticSchema)
        assert np.array_equal(schema.edges, bins)

    def test_bin_array(self):
        bins = [[0, 1], [1, 2], [2, 3.2]]
        schema = build_schema(bins=bins)
        assert isinstance(schema, StaticSchema)
        assert np.array_equal(schema.bins, bins)

    def test_wrong_arrays(self):
        bins = {
            "wrong_2nd_dim": [[4, 3, 4], [1, 2, 2]],
            "3dim": [[[2, 3], [3, 4], [5, 6]]],
        }
        for _, value in bins.items():
            with self.assertRaises(ValueError):
                _ = build_schema(bins=value)


class TestStaticSchema(unittest.TestCase):
    def test_no_bins_init(self):
        with self.assertRaises(ValueError) as err:
            _ = StaticSchema()
            assert err.msg == "Must specify either bins or edges."

    def test_bins_init(self):
        bins = [[0, 1], [1, 2], [2, 3]]
        edges = [0, 1, 2, 3]
        schema = StaticSchema(bins=bins)
        assert np.array_equal(bins, schema.bins)
        assert np.array_equal(edges, schema.edges)

    def test_edges_init(self):
        bins = [[0, 1], [1, 2], [2, 3]]
        edges = [0, 1, 2, 3]
        schema = StaticSchema(edges=edges)
        assert np.array_equal(bins, schema.bins)
        assert np.array_equal(edges, schema.edges)

    def test_bins_and_edges_init(self):
        bins = [[0, 1], [1, 2], [2, 3]]
        edges = [0, 1, 2, 3]
        with self.assertRaises(ValueError):
            _ = StaticSchema(bins=bins, edges=edges)

    def test_invalid_edges_shape(self):
        pass

    def test_invalid_bins_shape(self):
        pass


class TestNumpySchema(unittest.TestCase):
    pass


class TestFixedWidthSchema(unittest.TestCase):
    pass


class TestHumanSchema(unittest.TestCase):
    pass
