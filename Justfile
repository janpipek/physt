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

# Run pytest
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
    uv run --extra all basedpyright src/

# Run all the pre-commit checks on the whole code-base
pre-commit:
    uvx pre-commit run --all

# Create a wheel
build:
    rm -rf dist/
    uv build
    rm -rf src/physt.egg-info

# Publish to pypi.org
publish: build
    uv publish

# Run the project example in the CLI
examples:
    uv run --extra all python -m physt.examples

# Build the spinx documentation
docs:
    cd docs && uv run --extra all sphinx-apidoc -o . ../src/physt
    uv run --extra all sphinx-build -b html docs/ docs/_build/html
