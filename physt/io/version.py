from pkg_resources import parse_version

from packaging.version import Version

from physt import __version__

CURRENT_VERSION = __version__


class VersionError(Exception):
    pass


def require_compatible_version(compatible_version, word="File"):
    """Check that compatible version of input data is not too new."""
    if isinstance(compatible_version, str):
        compatible_version = parse_version(compatible_version)
    elif not isinstance(compatible_version, Version):
        raise ValueError("Type of `compatible_version` not understood.")

    current_version = parse_version(CURRENT_VERSION)
    if current_version < compatible_version:
        raise VersionError(
            f"{word} written for version >= {compatible_version}, this is {CURRENT_VERSION}."
        )
