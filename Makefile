venv_name = venv
venv_activate_path := ./$(venv_name)/bin/activate
package_name = neural_lifetimes
cov_args := --cov $(package_name) --cov-report=term-missing
not_slow = -m "not slow"

.PHONY: clean venv lint test slowtest cov slowcov docs

clean:
	rm -rf ./$(venv_name)

venv:
	python3 -m venv $(venv_name) ;\
	. $(venv_activate_path) ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install --upgrade -r requirements-dev.txt ;\
	pip install --upgrade -r requirements.txt

update:
	. $(venv_activate_path) ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install --upgrade -r requirements-dev.txt ;\
	pip install --upgrade -r requirements.txt

lint:
	. $(venv_activate_path) ;\
	flake8 $(package_name)/ ;\
	flake8 tests/

test:
	. $(venv_activate_path) ;\
	py.test $(not_slow) --disable-warnings

slowtest:
	. $(venv_activate_path) ;\
	py.test

cov:
	. $(venv_activate_path) ;\
	py.test $(cov_args) $(not_slow)

slowcov:
	. $(venv_activate_path) ;\
	py.test $(cov_args)

format:
	. $(venv_activate_path) ;\
	isort -rc . ;\
	autoflake -r --in-place --remove-unused-variables $(package_name)/ ;\
	autoflake -r --in-place --remove-unused-variables tests/ ;\
	black $(package_name)/ --skip-string-normalization ;\
	black tests/ --skip-string-normalization

checkformat:
	. $(venv_activate_path) ;\
	black $(package_name)/ --skip-string-normalization --check ;\
	black tests/ --skip-string-normalization --check

docs:
	. $(venv_activate_path) ;\
	cd docs/ ;\
	sphinx-apidoc -o . ../neural_lifetimes ;\
	sphinx-build -b html . build

publishdocs:
	cp -r docs/build/* pages/
