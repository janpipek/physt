from typing import Any

import pytest


@pytest.fixture()
def default_kwargs() -> dict[str, Any]:
    """Arguments to add to each plotting method."""
    return {}
