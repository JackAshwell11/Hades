"""Stores all the functionality which creates the game and makes it playable."""

from __future__ import annotations

# Builtin
from pathlib import Path

# Pip
import arcade

__author__ = "Aspect1103"
__license__ = "GNU GPLv3"
__version__ = "0.1.0"


# Add the resources directory to the resource loader
arcade.resources.add_resource_handle(
    "resources",
    Path(__file__).resolve().parent / "resources" / "textures",
)
