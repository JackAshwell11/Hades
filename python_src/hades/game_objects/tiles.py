"""Manages the different tiles that can exist in the game."""
from __future__ import annotations

# Custom
from hades.game_objects.objects import GameObject

__all__ = ("Wall", "Floor")


class Wall(GameObject):
    """Represents a wall tile in the game."""

    def __repr__(self: Wall) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Wall (Position={self.position})>"


class Floor(GameObject):
    """Represents a wall tile in the game."""

    def __repr__(self: Floor) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Floor (Position={self.position})>"
