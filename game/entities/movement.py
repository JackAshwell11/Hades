from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.entity import MOVEMENT_FORCE

# Get the logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from game.entities.base import Tile
    from game.entities.enemy import Enemy
    from game.vector_field import VectorField


class EnemyMovementManager:
    """
    Manages and processes logic needed for the enemy to move towards the player or
    wander around. This is a work in progress.

    Parameters
    ----------
    owner: Enemy
        The reference to the enemy object that controls this manager.
    """

    __slots__ = ("owner",)

    def __init__(self, owner: Enemy) -> None:
        self.owner: Enemy = owner

    def __repr__(self) -> str:
        return f"<EnemyMovement (Owner={self.owner})>"

    @property
    def vector_field(self) -> VectorField:
        """
        Gets the vector field for easy access.

        Returns
        -------
        VectorField
            The generated vector field.
        """
        # Make sure the vector field is valid
        assert self.owner.game.vector_field is not None

        # Get the vector field object
        return self.owner.game.vector_field

    @property
    def current_tile(self) -> Tile:
        """
        Gets the enemy's current tile.

        Returns
        -------
        Tile
            The enemy's current tile.
        """
        return self.vector_field.get_tile_at_position(*self.owner.tile_pos)

    def calculate_vector_field_force(self) -> tuple[float, float]:
        """
        Calculates the force to apply to an enemy which is using the vector field.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy.
        """
        # Get the vector direction the enemy needs to travel in
        vector_direction = self.vector_field.get_vector_direction(self.current_tile)

        # Calculate the force to apply to the enemy and return it
        return (
            vector_direction[0] * MOVEMENT_FORCE,
            vector_direction[1] * MOVEMENT_FORCE,
        )

    def calculate_wander_force(self) -> tuple[float, float]:
        """
        Calculates the force to apply to an enemy who is wandering. This currently does
        not work.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy.
        """
        return 0, 0
