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

        

