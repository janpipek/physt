from typing import Any, Dict

import pytest


@pytest.fixture()
def default_kwargs() -> Dict[str, Any]:
    """Arguments to add to each plotting method."""
    return {}
