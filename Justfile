@_:
    just --list

# Run the full tests suite via nox
nox:
    uvx --with "nox>=2021.10.9" nox

# Bump the version from x.y.z to x.(y+1).0
bump-minor:
    uvx bumpver update --tag final --minor

# Bump the version from x.y.z to x.y.(z+1)
bump-patch:
    uvx bumpver update --tag final --patch

[group('qa')]
test:
    uv run --extra all pytest

# Test typing with mypy (we want this to succeed)
[group('qa')]
mypy:
    uv run --extra all mypy src/ tests/

# Optionally test with pyright (we don't aim yet)
[group('qa')]
pyright:
    uv run --python 3.12 --extra all --with pyright pyright

# Run all the pre-commit checks on the whole code-base
pre-commit:
    uvx pre-commit run --all

build:
    rm -rf dist/
    uv build
    rm -rf src/physt.egg-info

publish: build
    uv publish

examples:
    uv run --extra all python -m physt.examples
