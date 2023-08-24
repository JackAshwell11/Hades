"""Manages the components and their required data for each game object."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

# Custom
from hades.game_objects.base import (
    AttackAlgorithms,
    ComponentBase,
    SteeringMovementState,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from hades.game_objects.base import ComponentType, SteeringBehaviours
    from hades.game_objects.steering import Vec2d

__all__ = (
    "Armour",
    "ArmourRegen",
    "ArmourRegenCooldown",
    "Attacks",
    "FireRatePenalty",
    "Footprint",
    "GameObjectAttributeBase",
    "Health",
    "InstantEffects",
    "Inventory",
    "KeyboardMovement",
    "Money",
    "MovementForce",
    "StatusEffect",
    "StatusEffects",
    "SteeringMovement",
    "ViewDistance",
)


@dataclass(slots=True)
class StatusEffect:
    """Represents a status effect that can be applied to a game object attribute.

    Attributes:
        value: The value that should be applied to the game object temporarily.
        duration: The duration the status effect should be applied for.
        original_value: The original value of the game object attribute which is being
            changed.
        original_max_value: The original maximum value of the game object attribute
            which is being changed.
        time_counter: The time counter for the status effect.
    """

    value: float
    duration: float
    original_value: float
    original_max_value: float
    time_counter: float = field(init=False)

    def __post_init__(self: StatusEffect) -> None:
        """Initialise the object after its initial initialisation."""
        self.time_counter = 0


class GameObjectAttributeBase(ComponentBase):
    """The base class for all game object attributes.

    Attributes:
        level_limit: The level limit of the game object attribute.
        max_value: The maximum value of the game object attribute.
        current_level: The current level of the game object attribute.
        applied_status_effect: The status effect currently applied to the game object.
    """

    __slots__ = (
        "_value",
        "level_limit",
        "max_value",
        "current_level",
        "applied_status_effect",
    )

    # Class variables
    instant_effect: ClassVar[bool] = True
    maximum: ClassVar[bool] = True
    status_effect: ClassVar[bool] = True
    upgradable: ClassVar[bool] = True

    def __init__(
        self: GameObjectAttributeBase,
        initial_value: int,
        level_limit: int,
    ) -> None:
        """Initialise the object.

        Args:
            initial_value: The initial value of the game object attribute.
            level_limit: The level limit of the game object attribute.
        """
        self._value: float = initial_value
        self.max_value: float = initial_value if self.maximum else float("inf")
        self.level_limit = level_limit
        self.current_level: int = 0
        self.applied_status_effect: StatusEffect | None = None

    @property
    def value(self: GameObjectAttributeBase) -> float:
        """Get the game object attribute's value.

        Returns:
            The game object attribute's value.
        """
        return self._value

    @value.setter
    def value(self: GameObjectAttributeBase, new_value: float) -> None:
        """Set the game object attribute's value.

        Args:
            new_value: The new game object attribute's value.
        """
        self._value = max(min(new_value, self.max_value), 0)


class Armour(GameObjectAttributeBase):
    """Allows a game object to have an armour attribute."""


@dataclass(slots=True)
class ArmourRegen(ComponentBase):
    """Allows a game object to regenerate armour.

    Attributes:
        time_since_armour_regen: The time since the game object last regenerated armour.
    """

    time_since_armour_regen: float = 0


class ArmourRegenCooldown(GameObjectAttributeBase):
    """Allows a game object to have an armour regen cooldown attribute."""

    instant_effect: ClassVar[bool] = False
    maximum: ClassVar[bool] = False


@dataclass(slots=True)
class Attacks(ComponentBase):
    """Allows a game object to attack other game objects."""

    attacks: Sequence[AttackAlgorithms]
    attack_state: int = 0


class FireRatePenalty(GameObjectAttributeBase):
    """Allows a game object to have a fire rate penalty attribute."""

    instant_effect: ClassVar[bool] = False
    maximum: ClassVar[bool] = False


@dataclass(slots=True)
class Footprint(ComponentBase):
    """Allows a game object to periodically leave footprints around the game map.

    Attributes:
        footprints: The footprints created by the game object.
        time_since_last_footprint: The time since the game object last left a footprint.
    """

    footprints: list[Vec2d] = field(default_factory=list)
    time_since_last_footprint: float = 0


class Health(GameObjectAttributeBase):
    """Allows a game object to have a health attribute."""


@dataclass(slots=True)
class InstantEffects(ComponentBase):
    """Allows a game object to provide instant effects.

    Attributes:
        level_limit: The level limit of the instant effects.
        instant_effects: The instant effects provided by the game object.
    """

    level_limit: int
    instant_effects: Mapping[ComponentType, Callable[[int], float]] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class Inventory(ComponentBase):
    """Allows a game object to have a fixed size inventory.

    Attributes:
        width: The width of the inventory.
        height: The height of the inventory.
        inventory: The game object's inventory.
    """

    width: int
    height: int
    inventory: list[int] = field(default_factory=list)


@dataclass(slots=True)
class KeyboardMovement(ComponentBase):
    """Allows a game object's movement to be controlled by the keyboard.

    Attributes:
        north_pressed: Whether the game object is moving north or not.
        south_pressed: Whether the game object is moving south or not.
        east_pressed: Whether the game object is moving east or not.
        west_pressed: Whether the game object is moving west or not.
    """

    north_pressed: bool = False
    south_pressed: bool = False
    east_pressed: bool = False
    west_pressed: bool = False


class Money(GameObjectAttributeBase):
    """Allows a game object to have a money attribute."""

    instant_effect: ClassVar[bool] = False
    maximum: ClassVar[bool] = False
    status_effect: ClassVar[bool] = False
    upgradable: ClassVar[bool] = False


class MovementForce(GameObjectAttributeBase):
    """Allows a game object to have a movement force attribute."""

    instant_effect: ClassVar[bool] = False
    maximum: ClassVar[bool] = False


@dataclass(slots=True)
class StatusEffects(ComponentBase):
    """Allows a game object to provide status effects.

    Attributes:
        level_limit: The level limit of the status effects.
        status_effects: The status effects provided by the game object.
    """

    level_limit: int
    status_effects: Mapping[
        ComponentType,
        tuple[Callable[[int], float], Callable[[int], float]],
    ] = field(default_factory=dict)


@dataclass(slots=True)
class SteeringMovement(ComponentBase):
    """Allows a game object's movement to be controlled by steering algorithms.

    Attributes:
        behaviours: The behaviours used by the game object.
        movement_state: The current movement state of the game object.
        target_id: The game object ID of the target.
        path_list: The list of points the game object should follow.
    """

    behaviours: Mapping[SteeringMovementState, Sequence[SteeringBehaviours]]
    movement_state: SteeringMovementState = SteeringMovementState.DEFAULT
    target_id: int = -1
    path_list: list[Vec2d] = field(default_factory=list)


class ViewDistance(GameObjectAttributeBase):
    """Allows a game object to have a view distance attribute."""

    instant_effect: ClassVar[bool] = False
    maximum: ClassVar[bool] = False
