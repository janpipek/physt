import unittest
from unittest.mock import Mock, patch

import numpy as np

from physt.schema import build_schema


class TestBuildSchema(unittest.TestCase):
    def test_numpy(self):
        from physt.schema import NumpySchema
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

    def test_string(self):
        from physt.schema import NumpySchema, UnknownBinCountAlgorithmError

        algo = "sqrt"
        schema = build_schema(bins=algo)
        assert isinstance(schema, NumpySchema)
        assert schema.bin_arg == algo

        algo = "nonexistent"
        with self.assertRaises(UnknownBinCountAlgorithmError):
            schema = build_schema(bins=algo)     

    def test_edge_array(self):
        from physt.schema import StaticSchema

        bins = [0, 1, 2, 3]
        schema = build_schema(bins=bins)
        assert isinstance(schema, StaticSchema)
        assert np.array_equal(schema.edges, bins)

    def test_bin_array(self):
        from physt.schema import StaticSchema

        bins = [[0, 1], [1, 2], [2, 3.2]]
        schema = build_schema(bins=bins)
        assert isinstance(schema, StaticSchema)
        assert np.array_equal(schema.bins, bins)

    def test_wrong_arrays(self):
        bins = {
            "wrong_2nd_dim": [[4, 3, 4], [1, 2, 2]],
            "3dim": [[[2, 3], [3, 4], [5, 6]]],
        }
        for key, value in bins.items():
            with self.assertRaises(ValueError):
                schema = build_schema(bins = value)