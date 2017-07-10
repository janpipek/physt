"""Various utility functions to support physt implementation.

These functions are mostly general Python functions, not specific
for numerical computing, histogramming, etc.
"""


def all_subclasses(cls):
    """All subclasses of a class.

    From: http://stackoverflow.com/a/17246726/2692780
    """
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(all_subclasses(subclass))
    return tuple(subclasses)


def find_subclass(base, name):
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
