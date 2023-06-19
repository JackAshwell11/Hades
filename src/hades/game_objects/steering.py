"""Manages the different steering behaviours and their output."""
from __future__ import annotations

# Builtin
from abc import ABCMeta, abstractmethod

__all__ = (
    "SteeringBehaviourBase",
    "Align",
    "Arrive",
    "Evade",
    "Flee",
    "FollowPath",
    "ObstacleAvoidance",
    "Pursuit",
    "Seek",
    "Separation",
    "Wander",
)

# TODO: See if these behaviours can be simplified


class SteeringBehaviourBase(metaclass=ABCMeta):
    """The base class for all steering behaviours."""

    __slots__ = ()

    @abstractmethod
    def get_steering_force(
        self: SteeringBehaviourBase,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """


class Align(SteeringBehaviourBase):
    """Allows a game object to move with the same velocity as another game object."""

    def get_steering_force(
        self: Align,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Arrive(SteeringBehaviourBase):
    """Allows a game object to move towards another game object and stand still."""

    def get_steering_force(
        self: Arrive,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Evade(SteeringBehaviourBase):
    """Allows a game object to flee from another game object's predicted position."""

    def get_steering_force(
        self: Evade,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Face(SteeringBehaviourBase):
    """Allows a game object to continuously face another game object."""

    def get_steering_force(
        self: Face,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Flee(SteeringBehaviourBase):
    """Allows a game object to run away from another game object."""

    def get_steering_force(
        self: Flee,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class FollowPath(SteeringBehaviourBase):
    """Allows a game object to follow a pre-determined path."""

    def get_steering_force(
        self: FollowPath,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class ObstacleAvoidance(SteeringBehaviourBase):
    """Allows a game object to avoid obstacles in its path."""

    def get_steering_force(
        self: ObstacleAvoidance,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Pursuit(SteeringBehaviourBase):
    """Allows a game object to seek towards another game object's predicted position."""

    def get_steering_force(
        self: Pursuit,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Seek(SteeringBehaviourBase):
    """Allows a game object to move towards another game object."""

    def get_steering_force(
        self: Seek,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Separation(SteeringBehaviourBase):
    """Allows a game object to separate from other game objects."""

    def get_steering_force(
        self: Separation,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Wander(SteeringBehaviourBase):
    """Allows a game object to move in a random direction for a short period of time."""

    def get_steering_force(
        self: Wander,
        current_position: tuple[float, float],
        target_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Get the new steering force from this behaviour.

        Args:
            current_position: The current position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


"""
Plan for enemy AI:

Facts:
- Player should always be faster than the enemy (except for some enemy types).

If enemy is not in range of player and no nearby smells:
    Walk randomly around the dungeon avoiding other enemies and walls until a smell is
    found.

If enemy is not in range of player and nearby smell:
    Follow the smell trail avoiding other enemies and walls making sure to steer so path
    is direct.

If enemy in range of player:
    Follow the player with random movement and periodically attack them (plus some
    randomness in the attack interval).
"""

# TODO: Ideal idea:
#  Have Movements component and a game object can enable specific movements which are
#  combined for the final force (mainly for steering).
#  Similarly have Attacks component and a game object can enable specific attacks which
#  can be cycled.
#  Smells and AI related stuff should be global then a game object can enable the AI
#  component to "listen" to that stuff.
