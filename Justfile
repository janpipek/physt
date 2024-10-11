nox:
    # Run the full tests suite via nox
    uvx --with "nox>=2021.10.9" nox

bump-minor:
    # Bump the version from x.y.z to x.(y+1).0
    uvx bumpver update --tag final --minor

bump-patch:
    # Bump the version from x.y.z to x.y.(z+1)
    uvx bumpver update --tag final --patch

pytest:
    uv run --extra all pytest

mypy:
    # Test typing with mypy (we want this to succeed)
    uv run --extra all mypy src/ tests/

pyright:
    # Optionally test with pyright (we don't aim yet)
    uv run --python 3.12 --extra all --with pyright pyright

pre-commit:
    # Run all the pre-commit checks on the whole code-base
    uvx pre-commit run --all

build:
    rm -rf dist/
    uv build
    rm -rf src/physt.egg-info

publish: build
    uv publish

examples:
    uv run --extra all python -m physt.examples
