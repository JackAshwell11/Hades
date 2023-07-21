"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from hades.game_objects.system import ECS

__all__ = (
    "AttackAlgorithms",
    "GameObjectAttributeSectionType",
    "ComponentData",
    "ComponentType",
    "GameObjectComponent",
    "SteeringBehaviours",
    "SteeringMovementState",
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
    FOOTPRINTS = auto()
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

    ENDURANCE = {ComponentType.HEALTH, ComponentType.MOVEMENT_FORCE}
    DEFENCE = {ComponentType.ARMOUR, ComponentType.ARMOUR_REGEN_COOLDOWN}


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
