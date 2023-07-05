"""Manages the different movement algorithms available to the game objects."""
from __future__ import annotations

# Builtin
import math
import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

# Pip
from pymunk import Vec2d

# Custom
from hades.constants import (
    MAX_SEE_AHEAD,
    MAX_VELOCITY,
    PATH_POINT_RADIUS,
    SLOWING_RADIUS,
    SPRITE_SIZE,
    WANDER_CIRCLE_DISTANCE,
    WANDER_CIRCLE_RADIUS,
)
from hades.game_objects.attributes import MovementForce
from hades.game_objects.base import (
    ComponentType,
    GameObjectComponent,
    SteeringBehaviours,
)

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData
    from hades.game_objects.system import ECS

__all__ = ("KeyboardMovement", "MovementBase", "SteeringMovement", "SteeringObject")


@dataclass(slots=True)
class SteeringObject:
    """Stores various data about a game object for use in steering.

    game_object_id: The game object ID.
    position: The position of the game object.
    velocity: The velocity of the game object.
    path_list: The list of points the game object should follow.
    """

    game_object_id: int
    position: Vec2d
    velocity: Vec2d
    path_list: list[Vec2d]


class MovementBase(GameObjectComponent, metaclass=ABCMeta):
    """The base class for all movement algorithms."""

    __slots__ = ("movement_force",)

    # Class variables
    component_type: ComponentType = ComponentType.MOVEMENTS
    player_controlled: bool = False

    def __init__(
        self: MovementBase,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.movement_force: MovementForce = cast(
            MovementForce,
            self.system.get_component_for_game_object(
                self.game_object_id,
                ComponentType.MOVEMENT_FORCE,
            ),
        )

    @abstractmethod
    def calculate_force(self: MovementBase) -> Vec2d:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """


class KeyboardMovement(MovementBase):
    """Allows a game object's movement to be controlled by the keyboard."""

    __slots__ = (
        "north_pressed",
        "south_pressed",
        "east_pressed",
        "west_pressed",
    )

    # Class variables
    player_controlled: bool = True

    def __init__(
        self: KeyboardMovement,
        game_object_id: int,
        system: ECS,
        _: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
        """
        super().__init__(game_object_id, system, _)
        self.north_pressed: bool = False
        self.south_pressed: bool = False
        self.east_pressed: bool = False
        self.west_pressed: bool = False

    def calculate_force(self: KeyboardMovement) -> Vec2d:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """
        return self.movement_force.value * Vec2d(
            self.east_pressed - self.west_pressed,
            self.north_pressed - self.south_pressed,
        )

    def __repr__(self: KeyboardMovement) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<KeyboardMovement (North pressed={self.north_pressed}) (South"
            f" pressed={self.south_pressed}) (East pressed={self.east_pressed}) (West"
            f" pressed={self.west_pressed})>"
        )


class SteeringMovement(MovementBase):
    """Allows a game object's movement to be controlled by steering algorithms.

    Attributes:
        walls: The list of wall positions in the game. This is in the form of
        bottom-left, bottom-right, top-right, and top-left.
    """

    __slots__ = ("_behaviours", "_current_steering", "_target_steering", "walls")

    def __init__(
        self: SteeringMovement,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self._behaviours: list[SteeringBehaviours] = component_data[
            "steering_behaviours"
        ]
        self._current_steering: SteeringObject = (
            self.system.get_steering_object_for_game_object(self.game_object_id)
        )
        self._target_steering: SteeringObject = (
            self.system.get_steering_object_for_game_object(-1)
        )
        self.walls: list[tuple[float, float]] = []

    @staticmethod
    def _flee(current_position: Vec2d, target_position: Vec2d) -> Vec2d:
        """Allow a game object to run away from another game object.

        Args:
            current_position: The position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        return current_position - target_position

    @staticmethod
    def _seek(current_position: Vec2d, target_position: Vec2d) -> Vec2d:
        """Allow a game object to move towards another game object.

        Args:
            current_position: The position of the game object.
            target_position: The position of the target game object.

        Returns:
            The new steering force from this behaviour.
        """
        return target_position - current_position

    def _arrive(self: SteeringMovement) -> Vec2d:
        """Allow a game object to move towards another game object and stand still.

        Returns:
            The new steering force from this behaviour.
        """
        # Calculate a vector to the target and its length
        direction = self._target_steering.position - self._current_steering.position

        # Check if the game object is inside the slowing area
        if direction.length < SLOWING_RADIUS:
            return direction * (direction.length / SLOWING_RADIUS)
        return direction

    def _evade(self: SteeringMovement) -> Vec2d:
        """Allow a game object to flee from another game object's predicted position.

        Returns:
            The new steering force from this behaviour.
        """
        # Calculate the future position of the target based on their distance and steer
        # away from it. Higher distances will require more time to reach, so the future
        # position will be further away
        return self._flee(
            self._current_steering.position,
            (
                self._target_steering.position
                + self._target_steering.velocity
                * (
                    self._target_steering.position.get_distance(
                        self._current_steering.position,
                    )
                    / MAX_VELOCITY
                )
            ),
        )

    def _follow_path(self: SteeringMovement) -> Vec2d:
        """Allow a game object to follow a pre-determined path.

        Returns:
            The new steering force from this behaviour.
        """
        if (
            self._current_steering.position.get_distance(
                self._current_steering.path_list[0],
            )
            <= PATH_POINT_RADIUS
        ):
            self._current_steering.path_list.append(
                self._current_steering.path_list.pop(0),
            )
        return self._seek(
            self._current_steering.position,
            self._current_steering.path_list[0],
        )

    def _obstacle_avoidance(self: SteeringMovement) -> Vec2d:
        """Allow a game object to avoid obstacles in its path.

        Returns:
            The new steering force from this behaviour.
        """
        # TODO: This still doesn't provide good obstacle avoidance. Maybe shelve it for
        #  now and work on the AI stuff

        def line_intersects_wall(obstacle: tuple[float, float]) -> bool:
            return (
                math.dist(obstacle, ahead) <= SPRITE_SIZE / 2
                or math.dist(obstacle, ahead_two) <= SPRITE_SIZE / 2
                or math.dist(obstacle, self._current_steering.position)
                <= SPRITE_SIZE / 2
            )

        # Create an ahead ray so the game object can detect obstacles ahead
        ahead, ahead_two = (
            self._current_steering.position
            + self._current_steering.velocity.normalized()
            * ((MAX_SEE_AHEAD * self._current_steering.velocity.length) / MAX_VELOCITY),
            self._current_steering.position
            + self._current_steering.velocity.normalized()
            * (
                (
                    (MAX_SEE_AHEAD * self._current_steering.velocity.length)
                    / MAX_VELOCITY
                )
                / 2
            ),
        )

        # Calculate the avoidance force to evade the most threatening obstacle for the
        # game object
        most_threatening = None
        for wall_position in self.walls:
            if line_intersects_wall(wall_position) and (
                most_threatening is None
                or math.dist(self._current_steering.position, wall_position)
                < math.dist(self._current_steering.position, most_threatening)
            ):
                most_threatening = Vec2d(*wall_position)
        if most_threatening is not None:
            return ahead - most_threatening
        return Vec2d(0, 0)

    def _pursuit(self: SteeringMovement) -> Vec2d:
        """Allow a game object to seek towards another game object's predicted position.

        Returns:
            The new steering force from this behaviour.
        """
        # Calculate the future position of the target based on their distance and steer
        # towards it. Higher distances will require more time to reach, so the future
        # position will be further away
        return self._seek(
            self._current_steering.position,
            self._target_steering.position
            + self._target_steering.velocity
            * (
                self._target_steering.position.get_distance(
                    self._current_steering.position,
                )
                / MAX_VELOCITY
            ),
        )

    def _wander(self: SteeringMovement) -> Vec2d:
        """Allow a game object to move in a random direction for a short period of time.

        Returns:
            The new steering force from this behaviour.
        """
        # Calculate the position of an invisible circle in front of the game object
        circle_center = (
            self._current_steering.velocity.normalized() * WANDER_CIRCLE_DISTANCE
        )

        # Add a displacement force to the centre of the circle to randomise the movement
        return circle_center + (Vec2d(0, -1) * WANDER_CIRCLE_RADIUS).rotated_degrees(
            random.randint(0, 360),
        )

    @property
    def target_id(self: SteeringMovement) -> int:
        """Get the target game object ID.

        Returns:
            The target game object ID.
        """
        return self._target_steering.game_object_id

    @target_id.setter
    def target_id(self: SteeringMovement, game_object_id: int) -> None:
        """Set the target game object ID.

        Args:
            game_object_id: The target game object ID.
        """
        if self._target_steering.game_object_id != game_object_id:
            self._target_steering = self.system.get_steering_object_for_game_object(
                game_object_id,
            )

    def calculate_force(self: SteeringMovement) -> Vec2d:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """
        steering_force = Vec2d(0, 0)
        for behaviour in self._behaviours:
            match behaviour:
                case SteeringBehaviours.ARRIVE:
                    steering_force += self._arrive()
                case SteeringBehaviours.EVADE:
                    steering_force += self._evade()
                case SteeringBehaviours.FLEE:
                    steering_force += self._flee(
                        self._current_steering.position,
                        self._target_steering.position,
                    )
                case SteeringBehaviours.FOLLOW_PATH:
                    steering_force += self._follow_path()
                case SteeringBehaviours.OBSTACLE_AVOIDANCE:
                    steering_force += self._obstacle_avoidance()
                case SteeringBehaviours.PURSUIT:
                    steering_force += self._pursuit()
                case SteeringBehaviours.SEEK:
                    steering_force += self._seek(
                        self._current_steering.position,
                        self._target_steering.position,
                    )
                case SteeringBehaviours.WANDER:
                    steering_force += self._wander()
        return self.movement_force.value * steering_force.normalized()

    def __repr__(self: SteeringMovement) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<SteeringMovement (Behaviour count={len(self._behaviours)}) (Target game"
            f" object ID={self.target_id})>"
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
