"""Manages the entities and their various functions."""
from __future__ import annotations

# Custom
from hades.game_objects.attributes import (
    ArmourMixin,
    ArmourRegenCooldownMixin,
    FireRatePenaltyMixin,
    HealthMixin,
    MoneyMixin,
)
from hades.game_objects.objects import GameObject

__all__ = ("Enemy", "Player")


class Player(
    HealthMixin,
    ArmourMixin,
    FireRatePenaltyMixin,
    ArmourRegenCooldownMixin,
    MoneyMixin,
    GameObject,
):
    """Represents the player character in the game."""

    def __init__(self: Player, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        kwargs: TODO
            The keyword arguments for the player.
        """
        super().__init__(**kwargs)

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

    def __init__(self: Enemy, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        kwargs: TODO
            The keyword arguments for the enemy.
        """
        super().__init__(**kwargs)

    def __repr__(self: Enemy) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Enemy (Position={self.position})>"
