from typing import Union

from packaging.version import Version, parse

from physt import __version__

CURRENT_VERSION = __version__


class VersionError(Exception):
    pass


def require_compatible_version(compatible_version: Union[str, Version], word="File") -> None:
    """Check that compatible version of input data is not too new."""
    if isinstance(compatible_version, str):
        compatible_version = parse(compatible_version)
    elif not isinstance(compatible_version, Version):
        raise ValueError("Type of `compatible_version` not understood.")

    current_version = parse(CURRENT_VERSION)
    if current_version < compatible_version:
        raise VersionError(
            f"{word} written for version >= {compatible_version}, this is {CURRENT_VERSION}."
        )
