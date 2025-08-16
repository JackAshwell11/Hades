"""Stores all the functionality which creates the game and makes it playable."""

from __future__ import annotations

# Builtin
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import Final

# Pip
from arcade import get_window
from arcade.gui import UIFlatButton
from arcade.resources import add_resource_handle
from arcade.types import Color

__all__ = (
    "BUTTON_WIDTH",
    "UI_BACKGROUND_COLOUR",
    "UI_PADDING",
    "BackButton",
    "SceneType",
)


class SceneType(Enum):
    """Represents the different scenes in the game."""

    GAME = auto()
    GAME_OPTIONS = auto()
    INVENTORY = auto()
    LOAD_GAME = auto()
    SHOP = auto()
    START_MENU = auto()


# The size of the padding around the UI elements
UI_PADDING: Final[int] = 4

# The width of the buttons
BUTTON_WIDTH: Final[int] = 200

# The background colour of the UI
UI_BACKGROUND_COLOUR: Final[Color] = Color(198, 198, 198)


class BackButton(UIFlatButton):
    """Represents a button that returns to the previous view when clicked."""

    def __init__(self: BackButton) -> None:
        """Initialise the object."""
        super().__init__(text="Back", width=BUTTON_WIDTH)
        self.on_click = partial(  # type: ignore[method-assign]
            get_window().dispatch_event,
            "on_previous_view",
        )


# Add the resources directory to the resource loader
add_resource_handle("resources", Path(__file__).resolve().parent / "resources")
