from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants import ENEMY_MOVEMENT_FORCE

if TYPE_CHECKING:
    from entities.base import Entity
    from entities.player import Player


class AIBase:
    """The base class for all AI algorithms. These could either be movement or attacking
    algorithms (or both)."""

    def __init__(self) -> None:
        self.owner: Entity | None = None

    def __repr__(self) -> str:
        return f"<AIBase (Owner={self.owner})>"

    def calculate_movement(
        self, player: Player, walls: arcade.SpriteList
    ) -> tuple[float, float]:
        """
        Calculates the new position for an enemy.

        Parameters
        ----------
        player: Player
            The player object to move towards.
        walls: arcade.SpriteList
            The wall tiles which block the enemy's vision.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy to move it towards the target.
        """
        raise NotImplementedError


class FollowLineOfSight(AIBase):
    """An algorithm which moves the enemy towards the target if the enemy has line of
    sight with the target, and the target is within the enemy's view distance."""

    def __repr__(self) -> str:
        return f"<FollowLineOfSight (Owner={self.owner})>"

    def calculate_movement(
        self, target: Entity, walls: arcade.SpriteList
    ) -> tuple[float, float]:
        """
        Calculates the new position for an enemy.

        Parameters
        ----------
        target: Entity
            The player target to move towards.
        walls: arcade.SpriteList
            The wall tiles which block the enemy's vision.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy to move it towards the target.
        """
        # Make sure variables needed are valid
        assert self.owner is not None

        # Calculate the velocity for the enemy to move towards the player
        return (
            -(self.owner.center_x - target.center_x) * ENEMY_MOVEMENT_FORCE,
            -(self.owner.center_y - target.center_y) * ENEMY_MOVEMENT_FORCE,
        )
