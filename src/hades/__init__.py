"""Stores all the functionality which creates the game and makes it playable."""

from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import Final

# Pip
import pygame


class ViewType(Enum):
    """Represents the different views in the game."""

    START_MENU = auto()
    GAME = auto()
    PLAYER = auto()


# Initialise pygame
pygame.init()

# The size of the padding around the UI elements
UI_PADDING: Final[int] = 4
