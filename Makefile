# Platform constants
ifeq ($(OS), Windows_NT)
	PY = python
	VENV_SUBDIR = Scripts
	DEL = rmdir /Q /S
else
	PY = python3
	VENV_SUBDIR = bin
	DEL = rm -rf
endif

# Virtual environment name and its paths
VENV_NAME = .venv
export VENV_PATH := $(abspath ${VENV_NAME})
export PATH := ${VENV_PATH}/$(VENV_SUBDIR):${PATH}

.PHONY:  # Ensures this only runs if a virtual environment doesn't exist
${VENV_NAME}:
	poetry install --with build,pre-commit,test


# -------------------- Builds --------------------
basic-venv:  # Creates a virtual environment with only the required dependencies
	poetry install -vvv

fresh-venv:  # Creates a fresh virtual environment
	$(DEL) "$(VENV_PATH)"
	make full-venv

full-venv: ${VENV_NAME} # Creates a virtual environment with all dependencies installed

update-venv: ${VENV_NAME}  # Updates the poetry virtual environment
	poetry update -vvv

pre-commit: ${VENV_NAME}  # Runs pre-commit
	poetry run pre-commit run --all-files

test: ${VENV_NAME}  # Runs the tests using Pytest
	poetry run pytest

tox: ${VENV_NAME}  # Runs the entire test suite using Tox
	poetry run tox

build: ${VENV_NAME}  # Builds the game into an executable form
	poetry run python -m build

compile: ${VENV_NAME}  # Compiles the extensions so they can be accessed in Python
	poetry run python -m hades.extensions.compile
