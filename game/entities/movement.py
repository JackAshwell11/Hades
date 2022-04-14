from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from entities.base import Entity
    from entities.player import Player


# Get the logger
logger = logging.getLogger(__name__)


class AIMovementBase:
    """
    The base class for all AI enemy movement algorithms.

    Parameters
    ----------
    owner: Entity
        The owner of this AI algorithm.
    """

    def __init__(self, owner: Entity) -> None:
        self.owner: Entity = owner

    def __repr__(self) -> str:
        return f"<AIMovementBase (Owner={self.owner})>"

    def distance_to_player(self, player: Player) -> tuple[float, float]:
        """
        Calculates the distance between the enemy and the player

        Parameters
        ----------
        player: Player
            The player to calculate the distance to.

        Returns
        -------
        tuple[float, float]
            The distance to the player.
        """
        return (
            self.owner.center_x - player.center_x,
            self.owner.center_y - player.center_y,
        )

    def calculate_movement(self, player: Player) -> tuple[float, float]:
        """
        Calculates the force to apply to an enemy.

        Parameters
        ----------
        player: Player
            The player object.

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


class FollowLineOfSight(AIMovementBase):
    """
    An algorithm which moves the enemy towards the player if the enemy has line of
    sight with the player and the player is within the enemy's view distance.

    Parameters
    ----------
    owner: Entity
        The owner of this AI algorithm.
    """

    def __init__(self, owner: Entity) -> None:
        super().__init__(owner)

    def __repr__(self) -> str:
        return f"<FollowLineOfSight (Owner={self.owner})>"

    def calculate_movement(self, player: Player) -> tuple[float, float]:
        """
        Calculates the new position for an enemy based on the

        Parameters
        ----------
        player: Player
            The player to move towards.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy to move it towards the player.
        """
        # Make sure we have the movement force. This avoids a circular import
        from constants.general import MOVEMENT_FORCE

        # Calculate the velocity for the enemy to move towards the player
        distance: tuple[float, float] = self.distance_to_player(player)
        return (
            -distance[0] * MOVEMENT_FORCE,
            -distance[1] * MOVEMENT_FORCE,
        )


class Jitter(AIMovementBase):
    """
    An algorithm which .

    Parameters
    ----------
    owner: Entity
        The owner of this AI algorithm.
    """

    def __init__(self, owner: Entity) -> None:
        super().__init__(owner)

    def __repr__(self) -> str:
        return f"<Jitter (Owner={self.owner})>"

    def calculate_movement(self, player: Player) -> tuple[float, float]:
        """
        Calculates the new position for an enemy.

        Parameters
        ----------
        player: Player
            The player object. This is not used for this algorithm.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy.
        """
        raise NotImplementedError


class MoveAwayLineOfSight(AIMovementBase):
    """
    An algorithm which moves the enemy away from the player if the enemy has line of
    sight with the player and the player is within the enemy's view distance.

    Parameters
    ----------
    owner: Entity
        The owner of this AI algorithm.
    """

    def __init__(self, owner: Entity) -> None:
        super().__init__(owner)

    def __repr__(self) -> str:
        return f"<MoveAway (Owner={self.owner})>"

    def calculate_movement(self, player: Player) -> tuple[float, float]:
        """
        Calculates the new position for an enemy.

        Parameters
        ----------
        player: Player
            The player to move away from.

        Returns
        -------
        tuple[float, float]
            The calculated force to apply to the enemy to move it away from the player.
        """
        # Make sure we have the movement force. This avoids a circular import
        from constants.general import MOVEMENT_FORCE

        # Calculate the velocity for the enemy to move towards the player
        distance: tuple[float, float] = self.distance_to_player(player)
        return (
            distance[0] * MOVEMENT_FORCE,
            distance[1] * MOVEMENT_FORCE,
        )
