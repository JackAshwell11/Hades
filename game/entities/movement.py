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


class AIMovementBase:
    """
    The base class for all AI enemy movement algorithms.

    Parameters
    ----------
    owner: Enemy
        The owner of this AI algorithm.
    """

    __slots__ = ("owner",)

    def __init__(self, owner: Enemy) -> None:
        self.owner: Enemy = owner

    def __repr__(self) -> str:
        return f"<AIMovementBase (Owner={self.owner})>"

    def calculate_movement(self) -> tuple[float, float]:
        """
        Calculates the force to apply to an enemy.

        Raises
        ------
        NotImplementedError
            The function is not implemented.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy.
        """
        raise NotImplementedError


class VectorFieldMovement(AIMovementBase):
    """
    Simplifies the logic needed for the enemy to interact with the vector field and move
    around the game map.

    Parameters
    ----------
    owner: Enemy
        The owner of this movement algorithm.

    Attributes
    ----------
    target_tile: Tile
        f
    """

    __slots__ = ("target_tile",)

    def __init__(self, owner: Enemy) -> None:
        super().__init__(owner)
        self.target_tile: Tile = self.owner.game.tile_sprites[-1]

    def __repr__(self) -> str:
        return f"<VectorFieldMovement (Owner={self.owner})>"

    @property
    def tile_pos(self) -> tuple[int, int]:
        """
        Gets the enemy's current tile position.

        Returns
        -------
        tuple[int, int]
            The enemy's current tile position.
        """
        return self.owner.tile_pos

    @property
    def current_tile(self) -> Tile:
        """
        Gets the enemy's current tile.

        Returns
        -------
        Tile
            The enemy's current tile.
        """
        return self.vector_field.get_tile_at_position(*self.tile_pos)

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

    def calculate_movement(self) -> tuple[float, float]:
        """
        Calculates the force to apply to an enemy.

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


class WanderMovement(AIMovementBase):
    """
    An algorithm which .

    Parameters
    ----------
    owner: Enemy
        The owner of this movement algorithm.
    """

    __slots__ = ()

    def __init__(self, owner: Enemy) -> None:
        super().__init__(owner)

    def __repr__(self) -> str:
        return f"<Wander (Owner={self.owner})>"

    def calculate_movement(self) -> tuple[float, float]:
        """
        Calculates the force to apply to an enemy.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy.
        """
