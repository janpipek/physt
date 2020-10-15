import os

FREE_ARITHMETICS: bool = os.environ.get("PHYST_FREE_ARITHMETICS", "0") == "1"
"""Whether to allow arithmetic operations regardless of their reasonability."""

del os