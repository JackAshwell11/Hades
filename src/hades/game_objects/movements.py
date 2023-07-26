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
    MAX_VELOCITY,
    PATH_POINT_RADIUS,
    SLOWING_RADIUS,
    TARGET_DISTANCE,
    WANDER_CIRCLE_DISTANCE,
    WANDER_CIRCLE_RADIUS,
)
from hades.game_objects.attributes import MovementForce
from hades.game_objects.base import (
    ComponentType,
    GameObjectComponent,
    SteeringBehaviours,
    SteeringMovementState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hades.game_objects.base import ComponentData
    from hades.game_objects.system import ECS

__all__ = (
    "KeyboardMovement",
    "MovementBase",
    "SteeringMovement",
    "SteeringObject",
    "flee",
    "seek",
    "arrive",
    "evade",
    "follow_path",
    "obstacle_avoidance",
    "pursuit",
    "wander",
)


@dataclass(slots=True)
class SteeringObject:
    """Stores various data about a game object for use in steering.

    game_object_id: The game object ID.
    position: The position of the game object.
    velocity: The velocity of the game object.
    """

    game_object_id: int
    position: Vec2d
    velocity: Vec2d


def flee(current_position: Vec2d, target_position: Vec2d) -> Vec2d:
    """Allow a game object to run away from another game object.

    Args:
        current_position: The position of the game object.
        target_position: The position of the target game object.

    Returns:
        The new steering force from this behaviour.
    """
    return current_position - target_position


def seek(current_position: Vec2d, target_position: Vec2d) -> Vec2d:
    """Allow a game object to move towards another game object.

    Args:
        current_position: The position of the game object.
        target_position: The position of the target game object.

    Returns:
        The new steering force from this behaviour.
    """
    return target_position - current_position


def arrive(current_position: Vec2d, target_position: Vec2d) -> Vec2d:
    """Allow a game object to move towards another game object and stand still.

    Args:
        current_position: The position of the game object.
        target_position: The position of the target game object.

    Returns:
        The new steering force from this behaviour.
    """
    # Calculate a vector to the target and its length
    direction = target_position - current_position

    # Check if the game object is inside the slowing area
    if direction.length < SLOWING_RADIUS:
        return direction * (direction.length / SLOWING_RADIUS)
    return direction


def evade(current_position: Vec2d, target_steering: SteeringObject) -> Vec2d:
    """Allow a game object to flee from another game object's predicted position.

    Args:
        current_position: The position of the game object.
        target_steering: The target's steering object.

    Returns:
        The new steering force from this behaviour.
    """
    # Calculate the future position of the target based on their distance and steer
    # away from it. Higher distances will require more time to reach, so the future
    # position will be further away
    return flee(
        current_position,
        target_steering.position
        + target_steering.velocity
        * (target_steering.position.get_distance(current_position) / MAX_VELOCITY),
    )


def follow_path(current_position: Vec2d, path_list: list[Vec2d]) -> Vec2d:
    """Allow a game object to follow a pre-determined path.

    Args:
        current_position: The position of the game object.
        path_list: The list of points the game object should follow.

    Returns:
        The new steering force from this behaviour.
    """
    if current_position.get_distance(path_list[0]) <= PATH_POINT_RADIUS:
        path_list.append(path_list.pop(0))
    return seek(current_position, path_list[0])


def obstacle_avoidance() -> Vec2d:
    """Allow a game object to avoid obstacles in its path.

    Returns:
        The new steering force from this behaviour.
    """
    # TODO: Implement this steering behaviour. Could use raycasts, but not sure how
    #  to implement them efficiently
    raise NotImplementedError


def pursuit(current_position: Vec2d, target_steering: SteeringObject) -> Vec2d:
    """Allow a game object to seek towards another game object's predicted position.

    Args:
        current_position: The position of the game object.
        target_steering: The target's steering object.

    Returns:
        The new steering force from this behaviour.
    """
    # Calculate the future position of the target based on their distance and steer
    # towards it. Higher distances will require more time to reach, so the future
    # position will be further away
    return seek(
        current_position,
        target_steering.position
        + target_steering.velocity
        * (target_steering.position.get_distance(current_position) / MAX_VELOCITY),
    )


def wander(current_velocity: Vec2d) -> Vec2d:
    """Allow a game object to move in a random direction for a short period of time.

    Args:
        current_velocity: The velocity of the game object.

    Returns:
        The new steering force from this behaviour.
    """
    # Calculate the position of an invisible circle in front of the game object
    circle_center = current_velocity.normalized() * WANDER_CIRCLE_DISTANCE

    # Add a displacement force to the centre of the circle to randomise the movement
    return circle_center + (Vec2d(0, -1) * WANDER_CIRCLE_RADIUS).rotated_degrees(
        random.randint(0, 360),
    )


class MovementBase(GameObjectComponent, metaclass=ABCMeta):
    """The base class for all movement algorithms.

    Attributes:
        movement_force: The game object's movement force component.
    """

    __slots__ = ("movement_force",)

    # Class variables
    component_type: ComponentType = ComponentType.MOVEMENTS
    is_player_controlled: bool = False

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
    """Allows a game object's movement to be controlled by the keyboard.

    Attributes:
        north_pressed: Whether the game object is moving north or not.
        south_pressed: Whether the game object is moving south or not.
        east_pressed: Whether the game object is moving east or not.
        west_pressed: Whether the game object is moving west or not.
    """

    __slots__ = (
        "north_pressed",
        "south_pressed",
        "east_pressed",
        "west_pressed",
    )

    # Class variables
    is_player_controlled: bool = True

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
        target_id: The game object ID of the target.
        walls: The list of wall positions in the game.
        path_list: The list of points the game object should follow.
    """

    __slots__ = (
        "_behaviours",
        "_movement_state",
        "_current_steering",
        "target_id",
        "walls",
        "path_list",
    )

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
        self._behaviours: Mapping[
            SteeringMovementState,
            Sequence[SteeringBehaviours],
        ] = component_data["steering_behaviours"]
        self._movement_state: SteeringMovementState = SteeringMovementState.DEFAULT
        self._current_steering: SteeringObject = (
            self.system.get_steering_object_for_game_object(self.game_object_id)
        )
        self.target_id: int = -1
        self.walls: list[tuple[float, float]] = []
        self.path_list: list[Vec2d] = []

    def calculate_force(self: SteeringMovement) -> Vec2d:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """
        # Determine if the movement state should change or not
        target_steering = self.system.get_steering_object_for_game_object(
            self.target_id,
        )
        if (
            math.dist(self._current_steering.position, target_steering.position)
            <= TARGET_DISTANCE
        ):
            self._movement_state = SteeringMovementState.TARGET
        elif self.path_list:
            self._movement_state = SteeringMovementState.FOOTPRINT
        else:
            self._movement_state = SteeringMovementState.DEFAULT

        # Calculate the new force to apply to the game object
        steering_force = Vec2d(0, 0)
        for behaviour in self._behaviours.get(self._movement_state, []):
            match behaviour:
                case SteeringBehaviours.ARRIVE:
                    steering_force += arrive(
                        self._current_steering.position, target_steering.position,
                    )
                case SteeringBehaviours.EVADE:
                    steering_force += evade(
                        self._current_steering.position, target_steering,
                    )
                case SteeringBehaviours.FLEE:
                    steering_force += flee(
                        self._current_steering.position, target_steering.position,
                    )
                case SteeringBehaviours.FOLLOW_PATH:
                    steering_force += follow_path(
                        self._current_steering.position, self.path_list,
                    )
                case SteeringBehaviours.OBSTACLE_AVOIDANCE:
                    steering_force += obstacle_avoidance()
                case SteeringBehaviours.PURSUIT:
                    steering_force += pursuit(
                        self._current_steering.position, target_steering,
                    )
                case SteeringBehaviours.SEEK:
                    steering_force += seek(
                        self._current_steering.position, target_steering.position,
                    )
                case SteeringBehaviours.WANDER:
                    steering_force += wander(self._current_steering.velocity)
        return self.movement_force.value * steering_force.normalized()

    def update_path_list(
        self: SteeringMovement,
        footprints: list[tuple[float, float]],
    ) -> None:
        """Update the path list for the game object to follow.

        Args:
            footprints: The list of footprints to follow.
        """
        # Get the closest footprint to the target and test if one exists
        closest_footprints = [
            footprint
            for footprint in footprints
            if math.dist(self._current_steering.position, footprint) <= TARGET_DISTANCE
        ]
        if not closest_footprints:
            self.path_list.clear()
            return

        # Get the closest footprint to the target and start following the footprints
        # from that footprint
        target_footprint = min(
            closest_footprints,
            key=lambda footprint: math.dist(
                self.system.get_steering_object_for_game_object(
                    self.target_id,
                ).position,
                footprint,
            ),
        )
        self.path_list = footprints[footprints.index(target_footprint) :]

    def __repr__(self: SteeringMovement) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<SteeringMovement (Behaviour count={len(self._behaviours)}) (Target game"
            f" object ID={self.target_id})>"
        )
