"""Package information."""

__author__ = "Jan Pipek"
__author_email__ = "jan.pipek@gmail.com"
__url__ = "https://github.com/janpipek/physt"


def __getattr__(name):
    from importlib.metadata import version

    if name == "__version__":
        return version("physt")
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
