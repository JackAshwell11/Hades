"""Stores all the functionality which creates the game and makes it playable."""

from __future__ import annotations

# Builtin
from enum import Enum, auto
from pathlib import Path
from typing import Final

# Pip
from arcade.resources import add_resource_handle
from arcade.types import Color

__all__ = ("UI_BACKGROUND_COLOUR", "UI_PADDING", "SceneType")


class SceneType(Enum):
    """Represents the different scenes in the game."""

    START_MENU = auto()
    GAME = auto()
    INVENTORY = auto()
    SHOP = auto()


# The size of the padding around the UI elements
UI_PADDING: Final[int] = 4

# The background colour of the UI
UI_BACKGROUND_COLOUR: Final[Color] = Color(198, 198, 198)

# Add the resources directory to the resource loader
add_resource_handle("resources", Path(__file__).resolve().parent / "resources")
