# Hades

[![Tests](https://github.com/JackAshwell11/Hades/actions/workflows/test.yaml/badge.svg)](https://github.com/JackAshwell11/Hades/actions/workflows/test.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/JackAshwell11/Hades/main.svg)](https://results.pre-commit.ci/latest/github/JackAshwell11/Hades/main)
[![Coverage Status](https://coveralls.io/repos/github/JackAshwell11/Hades/badge.svg?branch=main)](https://coveralls.io/github/JackAshwell11/Hades?branch=main)
[![GitHub](https://img.shields.io/github/license/JackAshwell11/Hades)](LICENSE)

A procedurally generated bullet-hell dungeon crawler made in Python using
Arcade.

## How to run

### Pre-compiled releases (preferred)

The preferred and easiest way to play Hades is by using the pre-compiled
releases on GitHub. To play the game this way, follow these steps:

1. Download and extract the latest release from [here](https://github.com/JackAshwell11/Hades/releases).
2. Run the `window.exe` file to play the game.

### Locally

Another way to play Hades is by cloning the repository and running it locally.
To play the game this way, follow these steps:

1. Clone the repository using `git clone
   https://github.com/JackAshwell11/Hades.git`.
2. Ensure [uv](https://github.com/astral-sh/uv) is installed.
3. Run `uv venv` and `uv sync` to install the dependencies needed to run the
   game. This will create a virtual environment in the `hades/.venv` directory.
4. Run the command `uv run python src/hades/window.py` to play the game.

While this way is more convoluted and unstable, it will allow you to access the
latest version of the game with the newest features.

## Building the game

You can also compile the game locally if you choose. To do so, follow these
steps:

1. Clone the repository using `git clone
   https://github.com/JackAshwell11/Hades.git`.
2. Ensure [uv](https://github.com/astral-sh/uv) is installed.
3. Run `uv venv` and `uv sync` to install the dependencies needed to run the
   game. This will create a virtual environment in the `hades/.venv` directory.

## Contributing

See [here](.github/CONTRIBUTING.md) for more details.
