from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Optional, Tuple

# Pip
import arcade

# Custom
from constants import ENEMY_MOVEMENT_FORCE, ENEMY_VIEW_DISTANCE, SPRITE_WIDTH

if TYPE_CHECKING:
    from entities.entity import Entity


class FollowLineOfSight:
    """An algorithm which moves the enemy towards the target if the enemy has line of
    sight with the target, and the target is within the enemy's view distance."""

    def __init__(self) -> None:
        self.owner: Optional[Entity] = None

    def __repr__(self) -> str:
        return f"<FollowLineOfSight (Owner={self.owner})>"

    def calculate_movement(
        self, target: Entity, walls: arcade.SpriteList
    ) -> Tuple[float, float]:
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
        Tuple[float, float]
            The calculated force to apply to the enemy to move it towards the target.
        """
        assert self.owner is not None

        # Find if the enemy is within line of sight of the player and within view
        # distance
        if arcade.has_line_of_sight(
            (self.owner.center_x, self.owner.center_y),
            (target.center_x, target.center_y),
            walls,
            ENEMY_VIEW_DISTANCE * SPRITE_WIDTH,
        ):
            # Calculate the distance and direction to the player
            horizontal, vertical = (
                self.owner.center_x - target.center_x,
                self.owner.center_y - target.center_y,
            )
            # Apply the movement force in the specific direction
            return -horizontal * ENEMY_MOVEMENT_FORCE, -vertical * ENEMY_MOVEMENT_FORCE
        # Enemy does not have line of sight and is not within range
        return 0, 0
