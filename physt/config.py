import contextlib
import os
import sys

if sys.version_info >= (3, 7):
    _USES_CONTEXT_VARS = True
    import contextvars
else:
    _USES_CONTEXT_VARS = False


class _Config:
    """Main configuration singleton object.

    In Python 3.7+, it uses contextvars to enable async/thread-safe.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls)
            return cls._instance
        raise ValueError("Config is a singleton object.")

    def _make_var(self, name, default):
        if _USES_CONTEXT_VARS:
            var = contextvars.ContextVar(name, default=default)
        else:
            var = default
        setattr(self, name, var)

    def _get_value(self, name):
        var = getattr(self, name)
        if _USES_CONTEXT_VARS:
            return var.get()
        return var

    def _set_value(self, name, value) -> None:
        if _USES_CONTEXT_VARS:
            var = getattr(self, name)
            return var.set(value)
        else:
            setattr(self, name, value)

    @contextlib.contextmanager
    def _change_value(self, name, value):
        if _USES_CONTEXT_VARS:
            token = getattr(self, name).set(value)
            try:
                yield
            finally:
                getattr(self, name).reset(token)
        else:
            original_value = getattr(self, name)
            try:
                setattr(self, name, value)
                yield
            finally:
                setattr(self, name, original_value)

    def __init__(self):
        self._make_var("_free_arithmetics", os.environ.get("PHYST_FREE_ARITHMETICS", "0") == "1")

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

del os
