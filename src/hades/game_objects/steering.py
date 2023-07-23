"""Manages the different steering behaviours and their output."""
from __future__ import annotations

# Builtin
import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

# Pip
from pymunk import Vec2d

# Custom
from hades.constants import (
    SLOWING_RADIUS,
    WANDER_CIRCLE_DISTANCE,
    WANDER_CIRCLE_RADIUS,
)

__all__ = (
    "SteeringBehaviourBase",
    "Align",
    "Arrive",
    "Evade",
    "SteeringObject",
    "Flee",
    "FollowPath",
    "ObstacleAvoidance",
    "Pursuit",
    "Seek",
    "Separation",
    "Wander",
)

# TODO: See if these behaviours can be simplified


@dataclass(init=True, repr=True, slots=True)
class SteeringObject:
    """Stores the position and velocity of a game object for use in steering.

    game_object_id: The game object ID.
    position: The position of the game object.
    velocity: The velocity of the game object.
    """

    game_object_id: int
    position: Vec2d
    velocity: Vec2d


class SteeringBehaviourBase(metaclass=ABCMeta):
    """The base class for all steering behaviours."""

    __slots__ = ()

    @abstractmethod
    def get_steering_force(
        self: SteeringBehaviourBase,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """


class Align(SteeringBehaviourBase):
    """Allows a game object to move with the same velocity as another game object."""

    def get_steering_force(
        self: Align,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Arrive(SteeringBehaviourBase):
    """Allows a game object to move towards another game object and stand still."""

    def get_steering_force(
        self: Arrive,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        # Calculate a vector to the target and its length
        direction = target.position - current.position
        distance = direction.length

        # Check if the game object is inside the slowing area
        if distance < SLOWING_RADIUS:
            return direction * (distance / SLOWING_RADIUS)
        return direction


class Evade(SteeringBehaviourBase):
    """Allows a game object to flee from another game object's predicted position."""

    def get_steering_force(
        self: Evade,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Flee(SteeringBehaviourBase):
    """Allows a game object to run away from another game object."""

    def get_steering_force(
        self: Flee,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        return current.position - target.position


class FollowPath(SteeringBehaviourBase):
    """Allows a game object to follow a pre-determined path."""

    def get_steering_force(
        self: FollowPath,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class ObstacleAvoidance(SteeringBehaviourBase):
    """Allows a game object to avoid obstacles in its path."""

    def get_steering_force(
        self: ObstacleAvoidance,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Pursuit(SteeringBehaviourBase):
    """Allows a game object to seek towards another game object's predicted position."""

    def get_steering_force(
        self: Pursuit,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Seek(SteeringBehaviourBase):
    """Allows a game object to move towards another game object."""

    def get_steering_force(
        self: Seek,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        return target.position - current.position


class Separation(SteeringBehaviourBase):
    """Allows a game object to separate from other game objects."""

    def get_steering_force(
        self: Separation,
        *,
        current: SteeringObject,
        target: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.
            target: The position and velocity of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        raise NotImplementedError


class Wander(SteeringBehaviourBase):
    """Allows a game object to move in a random direction for a short period of time."""

    def get_steering_force(
        self: Wander,
        *,
        current: SteeringObject,
        **_: SteeringObject,
    ) -> Vec2d:
        """Get the new steering force from this behaviour.

        Args:
            current: The position and velocity of the game object.

        Returns:
            The new steering force from this behaviour.
        """
        # Calculate the position of an invisible circle in front of the game object
        circle_center = current.velocity.normalized() * WANDER_CIRCLE_DISTANCE

        # Add a displacement force to the centre of the circle to randomise the movement
        return circle_center + (Vec2d(0, -1) * WANDER_CIRCLE_RADIUS).rotated_degrees(
            random.randint(0, 360),
        )


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
#  Similarly, have Attacks component and a game object can enable specific attacks which
#  can be cycled.
#  Smells and AI related stuff should be global then a game object can enable the AI
#  component to "listen" to that stuff.

# TODO: Having steering_behaviours key be collection of far, following, and near lists
#  could help to merge ai into steering movement
