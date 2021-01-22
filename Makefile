.PHONY: mypy doc

all: mypy

doc:
	$(MAKE) -C doc autodoc
	$(MAKE) -C doc html

black:
	black -l 100 physt tests

mypy:
	mypy --ignore-missing-imports `find . ! -path '*/\.*' -name '*.py'`