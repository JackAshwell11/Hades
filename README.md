# Hades

[![GitHub](https://img.shields.io/github/license/Aspect1103/Hades)](LICENSE)
[![Run tests](https://github.com/Aspect1103/Hades/actions/workflows/test.yaml/badge.svg)](https://github.com/Aspect1103/Hades/actions/workflows/test.yaml)
[![Coverage Status](https://coveralls.io/repos/github/Aspect1103/Hades/badge.svg?branch=master)](https://coveralls.io/github/Aspect1103/Hades?branch=master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Aspect1103/Hades.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Aspect1103/Hades/context:python)

A procedurally generated bullet-hell dungeon crawler made in Python using Arcade.

## How to run

### Pre-compiled releases (preferred)

The preferred and easiest way to play Hades is by using the pre-compiled releases on GitHub. To play the game this way, follow these steps:

1. Download and extract the latest release from [here](https://github.com/Aspect1103/Hades/releases).
2. Run the `window.exe` file to play the game.

### Locally

Another way to play Hades is by cloning the repository and running it locally. To play the game this way, follow these steps:

1. Clone the repository using `git clone https://github.com/Aspect1103/Hades.git`.
2. Ensure [Poetry](https://python-poetry.org/) is installed and virtual environments are created in the project directory using `poetry config virtualenvs.in-project true --local`.
3. Run `poetry install --no-dev` to install the dependencies needed to run the game.
4. Run the `window.py` file in the `hades` directory to play the game. Optionally, you can run `python hades` in the CLI (make sure the virtual environment is active first).

While this way is more convoluted and unstable, it will allow you to access the latest version of the game with the newest features.

## Building the game

You can also compile the game locally if you choose. To do so, follow these steps:

1. Clone the repository using `git clone https://github.com/Aspect1103/Hades.git`.
2. Ensure [Poetry](https://python-poetry.org/) is installed and virtual environments are created in the project directory using `poetry config virtualenvs.in-project true --local`.
3. Run `poetry install` to install the dependencies needed to build the game.
4. Either run the `build.py` file or run `python -m build.py` to build the game locally. Optionally, you can run `make build` if you have [Make](https://www.gnu.org/software/make/manual/make.html) installed.

## Contributing

Contributions are welcomed and greatly appreciated as they help to improve Hades and add more features. To start contributing, follow these steps:

1. Clone the repository using `git clone https://github.com/Aspect1103/Hades.git`.
2. Ensure [Poetry](https://python-poetry.org/) is installed and virtual environments are created in the project directory using `poetry config virtualenvs.in-project true --local`.
3. Run `poetry install` to install all the dependencies.
4. Run `pre-commit install` to install the pre-commit hooks.

Contributions should be made through [issues](https://github.com/Aspect1103/Hades/issues) and [pull requests](https://github.com/Aspect1103/Hades/pulls).
