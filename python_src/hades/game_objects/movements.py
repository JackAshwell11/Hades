"""Manages the different movement algorithms available."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hades.game_objects.objects import GameObject

__all__ = ()


class SteeringMixin:
    """Allow a game object to steer around the game map."""

    def calculate_steering_force(self: type[GameObject]) -> tuple[float, float]:
        """TODO.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply.
        """
        raise NotImplementedError
