"""Helper functions"""
from collections.abc import Callable
import warnings

class deprecate(object):
    """Decorate a function to emit a DeprecationWarning."""

    def __init__(self, message: str=""):
        self.message = message
    
    def __call__(self, func: Callable) -> Callable: 
        """Emit DeprecationWarning and call the funcion."""

        def wrapped_func(*args, **kwargs):
            warnings.warn(self.message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
    
        return wrapped_func
