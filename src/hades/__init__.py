"""Stores all the functionality which creates the game and makes it playable."""

from __future__ import annotations

# Builtin
import os
from pathlib import Path

# Pip
import arcade
import pyglet

__author__ = "Aspect1103"
__license__ = "GNU GPLv3"
__version__ = "0.1.0"

# Test if we're running the tests on Linux in CI
if (
    os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("RUNNER_OS") == "Linux"
):  # pragma: no cover
    pyglet.options["headless"] = True


# Add the resources directory to the resource loader
arcade.resources.add_resource_handle(
    "resources",
    Path(__file__).resolve().parent / "resources" / "textures",
)
