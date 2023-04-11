"""Manages the entities and their various functions."""
from __future__ import annotations

# Builtin
import logging

# Custom
from hades.game_objects.components import Inventory
from hades.game_objects.objects import GameObject

__all__ = ("Player",)

# Get the logger
logger = logging.getLogger(__name__)


class Player(GameObject, Inventory):
    """Represents the player character in the game."""

    def __init__(self, x: int, y: int) -> None:
        """Initialise the object.

        Parameters
        ----------
        x: int
            The x position of the player in the game map.
        y: int
            The y position of the player in the game map.
        """
        super().__init__(x, y)

    def __repr__(self) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Player (Position={self.position})>"


class Enemy(GameObject):
    """Represents the enemy character in the game."""

    def __init__(self, x: int, y: int) -> None:
        """Initialise the object.

        Parameters
        ----------
        x: int
            The x position of the enemy in the game map.
        y: int
            The y position of the enemy in the game map.
        """
        super().__init__(x, y)

    def __repr__(self) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Enemy (Position={self.position})>"
