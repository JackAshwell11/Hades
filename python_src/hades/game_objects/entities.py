"""Manages the entities and their various functions."""
from __future__ import annotations

# Custom
from hades.game_objects.components import Inventory
from hades.game_objects.objects import GameObject

__all__ = ("Player",)


class Player(GameObject, Inventory):
    """Represents the player character in the game."""

    def __repr__(self: Player) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Player (Position={self.position})>"


class Enemy(GameObject):
    """Represents the enemy character in the game."""

    def __repr__(self: Enemy) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Enemy (Position={self.position})>"
