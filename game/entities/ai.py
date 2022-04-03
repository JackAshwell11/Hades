from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING, Any

# Custom
from constants.entity import MOVEMENT_FORCE

if TYPE_CHECKING:
    import arcade
    from entities.base import Entity
    from entities.player import Player


# Get the logger
logger = logging.getLogger(__name__)


class AIMovementBase:
    """The base class for all AI movement algorithms."""

    def __init__(self) -> None:
        self.owner: Entity | None = None

    def __repr__(self) -> str:
        return f"<AIMovementBase (Owner={self.owner})>"

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


class AIAttackBase:
    """The base class for all AI attacking algorithms."""

    def __init__(self) -> None:
        self.owner: Entity | None = None

    def __repr__(self) -> str:
        return f"<AIAttackBase (Owner={self.owner})>"

    def process_attack(self, *args: tuple[Any, ...]) -> None:
        """
        Processes an attack for an enemy.

        Parameters
        ----------
        args: tuple[any, ...]
            A tuple with any number of parameters of any value. This means we could have
            any number of parameters (or even zero).
        """
        raise NotImplementedError


class FollowLineOfSight(AIMovementBase):
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
            -(self.owner.center_x - target.center_x) * MOVEMENT_FORCE,
            -(self.owner.center_y - target.center_y) * MOVEMENT_FORCE,
        )
