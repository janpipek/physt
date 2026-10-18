import contextlib
import contextvars
import os
from typing import Any


class _Config:
    """Main configuration singleton object.

    It uses contextvars to enable async/thread-safe.
    """

    _instance : "_Config | None" = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls)
            return cls._instance
        raise ValueError("Config is a singleton object.")

    def _make_var(self, name: str, default: Any):
        var = contextvars.ContextVar(name, default=default)
        setattr(self, name, var)

    def _get_value(self, name: str) -> Any:
        var = getattr(self, name)
        return var.get()

    def _set_value(self, name: str, value: Any) -> None:
        var = getattr(self, name)
        return var.set(value)

    @contextlib.contextmanager
    def _change_value(self, name: str, value: Any):
        token = getattr(self, name).set(value)
        try:
            yield
        finally:
            getattr(self, name).reset(token)

    def __init__(self):
        self._make_var(
            "_free_arithmetics", os.environ.get("PHYST_FREE_ARITHMETICS", "0") == "1"
        )

    @property
    def free_arithmetics(self) -> bool:
        """Whether to allow arithmetic operations regardless of their reasonability."""
        return self._get_value("_free_arithmetics")

    @free_arithmetics.setter
    def free_arithmetics(self, value: bool):
        self._set_value("_free_arithmetics", value)

    @contextlib.contextmanager
    def enable_free_arithmetics(self, value: bool = True):
        """Temporarily allow/disallow benevolent arithmetics rules."""
        with self._change_value("_free_arithmetics", value):
            yield


config = _Config()


__all__ = ["config"]
