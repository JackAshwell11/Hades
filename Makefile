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

# Nuitka constants
BASE_PATH := $(VENV_PATH)/../game
NUITKA_PATH := $(BASE_PATH)/window.py


# -------------------- Builds --------------------
prepare-venv: ${VENV_NAME}  # Creates a virtual environment if one doesn't exist

fresh-venv:  # Creates a fresh virtual environment
	$(DEL) "$(VENV_PATH)"
	make prepare-venv

pre-commit: ${VENV_NAME}  # Runs pre-commit
	pre-commit run --all-files

test: ${VENV_NAME}  # Runs the tests using Tox
	poetry run tox

build: ${VENV_NAME}  # Builds the game with nuitka
	nuitka "$(NUITKA_PATH)"\
 	--standalone\
 	--follow-imports\
 	--enable-plugin=numpy\
 	--include-data-dir=game/resources=game/resources
