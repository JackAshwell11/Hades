# Platform constants
ifeq ($(OS), Windows_NT)
	VENV_SUBDIR = Scripts
	DEL = rmdir /Q /S
else
	VENV_SUBDIR = bin
	DEL = rm -rf
endif

# Virtual environment name and its paths
VENV_NAME = .venv
export VENV_PATH := $(abspath ${VENV_NAME})
export PATH := ${VENV_PATH}/$(VENV_SUBDIR):${PATH}

.PHONY:  # Ensures this only runs if a virtual environment doesn't exist
${VENV_NAME}:
	poetry install


# -------------------- Builds --------------------
basic-venv:  # Creates a virtual environment with only the required dependencies
	poetry install -vvv

fresh-venv:  # Creates a fresh virtual environment
	$(DEL) "$(VENV_PATH)"
	make full-venv

full-venv: ${VENV_NAME} # Creates a virtual environment with all dependencies installed

update-venv: ${VENV_NAME}  # Updates the poetry virtual environment
	poetry update -vvv

test: ${VENV_NAME}  # Runs the tests using Pytest
	poetry run pytest

pre-commit: ${VENV_NAME}  # Runs pre-commit
	pre-commit run --all-files

tox: ${VENV_NAME}  # Runs the entire test suite using Tox
	tox

ruff: ${VENV_NAME}  # Runs Ruff on the entire project in fix mode
	ruff --fix .

executable: ${VENV_NAME}  # Builds the game into an executable form
	python -m build --executable

cpp: ${VENV_NAME}  # Compiles the C++ extensions and installs them into the virtual environment
	python -m build --cpp
