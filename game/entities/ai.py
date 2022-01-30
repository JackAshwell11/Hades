from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Optional

# Pip
import arcade

if TYPE_CHECKING:
    from .entity import Entity


class FollowLineOfSight:
    """An algorithm which moves the enemy towards the target if the enemy has line of
    sight with the target, and the target is within the enemy's view distance."""

    def __init__(self) -> None:
        self.owner: Optional[Entity] = None

    def calculate_movement(self, target: Entity, walls: arcade.SpriteList):
        """
        Calculates the new position for an enemy.

        Parameters
        ----------
        target: Entity
            The player target to move towards.
        walls: arcade.SpriteList
            The wall tiles to avoid.
        """
