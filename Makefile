.PHONY: mypy

all: mypy

black:
	black physt tests

mypy:
	mypy --ignore-missing-imports `find . ! -path '*/\.*' -name '*.py'`