.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

NAME = train_pybird_emulators

help:
	@echo
	@echo "style-check - check code with flake8"
	@echo "reformat    - runs black to reformat code"
	@echo "test        - run tests quickly with the default Python"
	@echo "test-all    - run tests on every Python version with tox"
	@echo "coverage    - check code coverage quickly with the default Python"
	@echo "docs        - generate Sphinx HTML documentation, including API docs"
	@echo "sdist       - cresate source package file in dist/ folder"
	@echo "publish     - create and publish package"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

reformat:
	isort --treat-all-comment-as-code src/$(NAME) tests
	ruff format --line-length 88 src/$(NAME) tests

style-check:
	ruff check --select E,F,W --ignore E501,E203 --line-length 88 src/$(NAME) tests/

test:
	py.test tests

coverage:
	coverage run --source src/$(NAME) -m pytest tests
	coverage report -m
	coverage html
	open htmlcov/index.html

docs: open ?= open docs/_build/html/index.html
docs:
	rm -f docs/$(NAME).rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ src/$(NAME)
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	coverage run --source src/$(NAME) -m pytest tests
	coverage report -m
	coverage html
	cp -R htmlcov docs/_build/html
	$(open)

sdist: clean
	pip install build
	python -m build -s
	ls -l dist

publish: sdist
	twine check dist/* && twine upload dist/*
