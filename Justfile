nox:
    # Run the full tests suite via nox
    uvx nox

bump-minor:
    # Bump the version from x.y.z to x.(y+1).0
    uvx bumpver update --tag final --minor

bump-patch:
    # Bump the version from x.y.z to x.y.(z+1)
    uvx bumpver update --tag final --patch

mypy:
    # Test typing with mypy (we want this to succeed)
    uv run mypy src/ tests/

pyright:
    # Optionally test with pyright (we don't aim yet)
    uv run --with pyright pyright
