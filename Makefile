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
	poetry install


# -------------------- Builds --------------------
prepare-venv: ${VENV_NAME}  # Creates a virtual environment if one doesn't exist

fresh-venv:  # Creates a fresh virtual environment
	$(DEL) "$(VENV_PATH)"
	make prepare-venv

update-venv: ${VENV_NAME}  # Updates the poetry virtual environment
	poetry update

pre-commit: ${VENV_NAME}  # Runs pre-commit
	pre-commit run --all-files

test: ${VENV_NAME}  # Runs the tests using Tox
	tox

build: ${VENV_NAME}  # Builds the game into an executable form
	python build.py
