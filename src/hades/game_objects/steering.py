"""Manages the different steering algorithms available to the components."""
from __future__ import annotations

# Builtin
import math
from dataclasses import dataclass

# Custom
from hades.constants import (
    MAX_SEE_AHEAD,
    MAX_VELOCITY,
    OBSTACLE_AVOIDANCE_ANGLE,
    PATH_POINT_RADIUS,
    SLOWING_RADIUS,
    SPRITE_SIZE,
    WANDER_CIRCLE_DISTANCE,
    WANDER_CIRCLE_RADIUS,
)

__all__ = (
    "KinematicObject",
    "Vec2d",
    "arrive",
    "evade",
    "flee",
    "follow_path",
    "obstacle_avoidance",
    "pursuit",
    "seek",
    "wander",
)


class Vec2d:
    """Represents a 2D vector."""

    __slots__ = ("x", "y")

    def __init__(self: Vec2d, x: float, y: float) -> None:
        """Initialise the object.

        Args:
            x: The x value of the vector.
            y: The y value of the vector.
        """
        self.x: float = x
        self.y: float = y

    def normalised(self: Vec2d) -> Vec2d:
        """Normalise the vector.

        Returns:
            The normalised vector.
        """
        if magnitude := abs(self):
            return Vec2d(self.x / magnitude, self.y / magnitude)
        return Vec2d(0, 0)

    def rotated(self: Vec2d, angle: float) -> Vec2d:
        """Rotate the vector by an angle.

        Args:
            angle: The angle to rotate the vector by in radians.

        Returns:
            The rotated vector.
        """
        sine, cosine = math.sin(angle), math.cos(angle)
        return Vec2d(
            self.x * cosine - self.y * sine,
            self.x * sine + self.y * cosine,
        )

    def get_angle_between(self: Vec2d, other: Vec2d) -> float:
        """Get the angle between this vector and another vector.

        This will always be between 0 and 2π.

        Args:
            other: The vector to get the angle to.

        Returns:
            The angle between this vector and the other vector.
        """
        return (
            math.atan2(
                self.x * other.y - self.y * other.x,
                self.x * other.x + self.y * other.y,
            )
            + 2 * math.pi
        ) % (2 * math.pi)

    def get_distance_to(self: Vec2d, other: Vec2d) -> float:
        """Get the distance to another vector.

        Args:
            other: The vector to get the distance to.

        Returns:
            The distance to the other vector.
        """
        return abs(self - other)

    def __add__(self: Vec2d, other: Vec2d) -> Vec2d:
        """Add another vector to this vector.

        Args:
            other: The vector to add to this vector.

        Returns:
            The result of the addition.
        """
        return Vec2d(self.x + other.x, self.y + other.y)

    def __sub__(self: Vec2d, other: Vec2d) -> Vec2d:
        """Subtract another vector from this vector.

        Args:
            other: The vector to subtract from this vector.

        Returns:
            The result of the subtraction.
        """
        return Vec2d(self.x - other.x, self.y - other.y)

    def __mul__(self: Vec2d, other: float) -> Vec2d:
        """Multiply the vector by a scalar.

        Args:
            other: The scalar to multiply the vector by.

        Returns:
            The result of the multiplication.
        """
        return Vec2d(self.x * other, self.y * other)

    def __floordiv__(self: Vec2d, other: float) -> Vec2d:
        """Divide the vector by a scalar.

        Args:
            other: The scalar to divide the vector by.

        Returns:
            The result of the division.
        """
        return Vec2d(self.x // other, self.y // other)

    def __abs__(self: Vec2d) -> float:
        """Return the absolute value of the vector.

        Returns:
            The absolute value of the vector.
        """
        return math.sqrt(self.x**2 + self.y**2)

    def __eq__(self: Vec2d, other: object) -> bool:
        """Check if this vector is equal to another vector.

        Args:
            other: The vector to check equality with.

        Returns:
            Whether the vectors are equal.
        """
        if not isinstance(other, Vec2d):  # pragma: no cover
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self: Vec2d) -> int:
        """Get the hash of this object.

        Returns:
            The hash of this object.
        """
        return hash((self.x, self.y))

    def __repr__(self: Vec2d) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Vec2d (X={self.x}) (Y={self.y})>"


@dataclass(slots=True)
class KinematicObject:
    """Stores various data about a game object for use in physics-related operations.

    position: The position of the game object.
    velocity: The velocity of the game object.
    rotation: The rotation of the game object.
    """

    position: Vec2d
    velocity: Vec2d
    rotation: float = 0


def flee(current_position: Vec2d, target_position: Vec2d) -> Vec2d:
    """Allow a game object to run away from another game object.

    Args:
        current_position: The position of the game object.
        target_position: The position of the target game object.

    Returns:
        The new steering force from this behaviour.
    """
    return (current_position - target_position).normalised()


def seek(current_position: Vec2d, target_position: Vec2d) -> Vec2d:
    """Allow a game object to move towards another game object.

    Args:
        current_position: The position of the game object.
        target_position: The position of the target game object.

    Returns:
        The new steering force from this behaviour.
    """
    return (target_position - current_position).normalised()


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
    if abs(direction) < SLOWING_RADIUS:
        return (direction * (abs(direction) / SLOWING_RADIUS)).normalised()
    return direction.normalised()


def evade(
    current_position: Vec2d,
    target_position: Vec2d,
    target_velocity: Vec2d,
) -> Vec2d:
    """Allow a game object to flee from another game object's predicted position.

    Args:
        current_position: The position of the game object.
        target_position: The position of the target game object.
        target_velocity: The velocity of the target game object.

    Returns:
        The new steering force from this behaviour.
    """
    # Calculate the future position of the target based on their distance and steer
    # away from it. Higher distances will require more time to reach, so the future
    # position will be further away
    return flee(
        current_position,
        target_position
        + target_velocity
        * (target_position.get_distance_to(current_position) / MAX_VELOCITY),
    )


def follow_path(current_position: Vec2d, path_list: list[Vec2d]) -> Vec2d:
    """Allow a game object to follow a pre-determined path.

    Args:
        current_position: The position of the game object.
        path_list: The list of points the game object should follow.

    Raises:
        IndexError: The path list is empty.

    Returns:
        The new steering force from this behaviour.
    """
    if current_position.get_distance_to(path_list[0]) <= PATH_POINT_RADIUS:
        path_list.append(path_list.pop(0))
    return seek(current_position, path_list[0])


def obstacle_avoidance(
    current_position: Vec2d,
    current_velocity: Vec2d,
    walls: set[Vec2d],
) -> Vec2d:
    """Allow a game object to avoid obstacles in its path.

    Returns:
        The new steering force from this behaviour.
    """

    def _raycast(position: Vec2d, velocity: Vec2d, angle: float = 0) -> Vec2d:
        """Cast a ray from the game object's position in the direction of its velocity.

        Args:
            position: The position of the game object.
            velocity: The velocity of the game object.
            angle: The angle to rotate the velocity by in radians.

        Returns:
            The point at which the ray collides with an obstacle. If this is -1, then
            there is no collision.
        """
        for point in (
            position + velocity.rotated(angle) * (step / 100)
            for step in range(int(SPRITE_SIZE), int(MAX_SEE_AHEAD), int(SPRITE_SIZE))
        ):
            if point // SPRITE_SIZE in walls:
                return point
        return Vec2d(-1, -1)

    # Check if the game object is going to collide with an obstacle
    forward_ray = _raycast(current_position, current_velocity)
    left_ray = _raycast(
        current_position,
        current_velocity,
        OBSTACLE_AVOIDANCE_ANGLE,
    )
    right_ray = _raycast(
        current_position,
        current_velocity,
        -OBSTACLE_AVOIDANCE_ANGLE,
    )

    # Check if there are any obstacles ahead
    if (
        forward_ray != Vec2d(-1, -1)
        and left_ray != Vec2d(-1, -1)
        and right_ray != Vec2d(-1, -1)
    ):
        # Turn around, there's a wall ahead
        return flee(current_position, forward_ray)
    if left_ray != Vec2d(-1, -1):
        # Turn right, there's a wall left
        return flee(current_position, left_ray)
    if right_ray != Vec2d(-1, -1):
        # Turn left, there's a wall right
        return flee(current_position, right_ray)

    # No obstacles ahead, move forward
    return Vec2d(0, 0)


def pursuit(
    current_position: Vec2d,
    target_position: Vec2d,
    target_velocity: Vec2d,
) -> Vec2d:
    """Allow a game object to seek towards another game object's predicted position.

    Args:
        current_position: The position of the game object.
        target_position: The position of the target game object.
        target_velocity: The velocity of the target game object.

    Returns:
        The new steering force from this behaviour.
    """
    # Calculate the future position of the target based on their distance and steer
    # towards it. Higher distances will require more time to reach, so the future
    # position will be further away
    return seek(
        current_position,
        target_position
        + target_velocity
        * (target_position.get_distance_to(current_position) / MAX_VELOCITY),
    )


def wander(current_velocity: Vec2d, displacement_angle: int) -> Vec2d:
    """Allow a game object to move in a random direction for a short period of time.

    Args:
        current_velocity: The velocity of the game object.
        displacement_angle: The angle of the displacement force in degrees.

    Returns:
        The new steering force from this behaviour.
    """
    # Calculate the position of an invisible circle in front of the game object
    circle_center = current_velocity.normalised() * WANDER_CIRCLE_DISTANCE

    # Add a displacement force to the centre of the circle to randomise the movement
    return (
        circle_center
        + (Vec2d(0, -1) * WANDER_CIRCLE_RADIUS).rotated(
            math.radians(displacement_angle),
        )
    ).normalised()
