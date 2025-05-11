"""Stores all the functionality which creates the game and makes it playable."""

from __future__ import annotations

# Builtin
from enum import Enum, auto
from pathlib import Path
from typing import Final

# Pip
import arcade


class ViewType(Enum):
    """Represents the different views in the game."""

    START_MENU = auto()
    GAME = auto()
    PLAYER = auto()


# The size of the padding around the UI elements
UI_PADDING: Final[int] = 4

# Add the resources directory to the resource loader
arcade.resources.add_resource_handle(
    "resources",
    Path(__file__).resolve().parent / "resources" / "textures",
)
