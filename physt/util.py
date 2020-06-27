"""Various utility functions to support physt implementation.

These functions are mostly general Python functions, not specific
for numerical computing, histogramming, etc.
"""
import warnings
from functools import wraps
from typing import Any, Dict, Tuple


def all_subclasses(cls: type) -> Tuple[type, ...]:
    """All subclasses of a class.

    From: http://stackoverflow.com/a/17246726/2692780
    """
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(all_subclasses(subclass))
    return tuple(subclasses)


def find_subclass(base: type, name: str) -> type:
    """Find a named subclass of a base class.

    Uses only the class name without namespace.
    """
    class_candidates = [klass
                        for klass in all_subclasses(base)
                        if klass.__name__ == name
                        ]
    if len(class_candidates) == 0:
        raise RuntimeError("No \"{0}\" subclass of \"{1}\".".format(base.__name__, name))
    elif len(class_candidates) > 1:
        raise RuntimeError("Multiple \"{0}\" subclasses of \"{1}\".".format(base.__name__, name))
    return class_candidates[0]


def pop_many(a_dict: Dict[str, Any], *args: str,  **kwargs) -> Dict[str, Any]:
    """Pop multiple items from a dictionary.
    
    Parameters
    ----------
    a_dict : Dictionary from which the items will popped
    args: Keys which will be popped (and not included if not present)
    kwargs: Keys + default value pairs (if key not found, this default is included)

    Returns
    -------
    A dictionary of collected items.
    """
    result = {}
    for arg in args:
        if arg in a_dict:
            result[arg] = a_dict.pop(arg)
    for key, value in kwargs.items():
        result[key] = a_dict.pop(key, value)
    return result


def deprecation_alias(f, deprecated_name: str):
    """Provide a deprecated copy of a function.

    Parameters
    ----------
    f : The correct function
    deprecated_name : The name the function will be given

    Examples
    --------
    >>> def new(x): return 1
    >>> old = deprecated_name(new, "old")
    """
    @wraps(f)
    def inner(*args, **kwargs):
        warnings.warn(
            f"{deprecated_name} is deprecated, use {f.__name__} instead",
            DeprecationWarning
        )
        return f(*args, **kwargs)
    return inner