"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
import math
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from hades.game_objects.system import ECS

__all__ = (
    "AttackAlgorithms",
    "GameObjectAttributeSectionType",
    "ComponentData",
    "ComponentType",
    "GameObjectComponent",
    "SteeringBehaviours",
    "SteeringMovementState",
    "Vec2d",
)


class AttackAlgorithms(Enum):
    """Stores the different types of attack algorithms available."""

    AREA_OF_EFFECT_ATTACK = auto()
    MELEE_ATTACK = auto()
    RANGED_ATTACK = auto()


class ComponentType(Enum):
    """Stores the different types of components available."""

    ARMOUR = auto()
    ARMOUR_REGEN = auto()
    ARMOUR_REGEN_COOLDOWN = auto()
    ATTACKS = auto()
    FIRE_RATE_PENALTY = auto()
    FOOTPRINT = auto()
    HEALTH = auto()
    INSTANT_EFFECTS = auto()
    INVENTORY = auto()
    MONEY = auto()
    MOVEMENTS = auto()
    MOVEMENT_FORCE = auto()
    STATUS_EFFECTS = auto()
    VIEW_DISTANCE = auto()


class GameObjectAttributeSectionType(Enum):
    """Stores the sections which group game object attributes together."""

    ENDURANCE: AbstractSet[ComponentType] = {
        ComponentType.HEALTH,
        ComponentType.MOVEMENT_FORCE,
    }
    DEFENCE: AbstractSet[ComponentType] = {
        ComponentType.ARMOUR,
        ComponentType.ARMOUR_REGEN_COOLDOWN,
    }


class SteeringBehaviours(Enum):
    """Stores the different types of steering behaviours available."""

    ARRIVE = auto()
    EVADE = auto()
    FLEE = auto()
    FOLLOW_PATH = auto()
    OBSTACLE_AVOIDANCE = auto()
    PURSUIT = auto()
    SEEK = auto()
    WANDER = auto()


class SteeringMovementState(Enum):
    """Stores the different states the steering movement component can be in."""

    DEFAULT = auto()
    FOOTPRINT = auto()
    TARGET = auto()


class ComponentData(TypedDict, total=False):
    """Holds the data needed to initialise the components.

    attributes: The data for the game object attributes.
    enabled_attacks: The attacks which are enabled for the game object.
    steering_behaviours: The steering behaviours to use.
    instant_effects: The instant effects that this game object can apply.
    inventory_size: The size of the game object's inventory.
    status_effects: The status effects that this game object can apply.
    """

    attributes: Mapping[ComponentType, tuple[int, int]]
    enabled_attacks: Sequence[AttackAlgorithms]
    steering_behaviours: Mapping[SteeringMovementState, Sequence[SteeringBehaviours]]
    instant_effects: tuple[int, Mapping[ComponentType, Callable[[int], float]]]
    inventory_size: tuple[int, int]
    status_effects: tuple[
        int,
        Mapping[ComponentType, tuple[Callable[[int], float], Callable[[int], float]]],
    ]


class GameObjectComponent:
    """The base class for all game object components."""

    __slots__ = ("game_object_id", "system")

    # Class variables
    component_type: ComponentType

    def __init__(
        self: GameObjectComponent,
        game_object_id: int,
        system: ECS,
        _: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
        """
        self.game_object_id: int = game_object_id
        self.system: ECS = system

    def on_update(self: GameObjectComponent, delta_time: float) -> None:
        """Process update logic.

        Args:
            delta_time: Time interval since the last time the function was called.
        """


class Vec2d(NamedTuple):
    """Represents a 2D vector.

    Attributes:
        x: The x value of the vector.
        y: The y value of the vector.
    """

    x: float
    y: float

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
        return math.atan2(
            self.x * other.y - self.y * other.x,
            self.x * other.x + self.y * other.y,
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

    def __repr__(self: Vec2d) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Vec2d (X={self.x}) (Y={self.y})>"
