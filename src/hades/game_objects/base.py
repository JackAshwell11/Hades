"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.system import ECS

__all__ = (
    "GameObjectAttributeSectionType",
    "ComponentData",
    "ComponentType",
    "GameObjectComponent",
    "ATTACK_ALGORITHMS",
)


class ComponentType(Enum):
    """Stores the different types of components available."""

    AREA_OF_EFFECT_ATTACK = auto()
    ARMOUR = auto()
    ARMOUR_REGEN = auto()
    ARMOUR_REGEN_COOLDOWN = auto()
    ATTACK_MANAGER = auto()
    FIRE_RATE_PENALTY = auto()
    HEALTH = auto()
    INSTANT_EFFECTS = auto()
    INVENTORY = auto()
    MELEE_ATTACK = auto()
    MONEY = auto()
    MOVEMENTS = auto()
    MOVEMENT_FORCE = auto()
    RANGED_ATTACK = auto()
    STATUS_EFFECTS = auto()
    VIEW_DISTANCE = auto()


class GameObjectAttributeSectionType(Enum):
    """Stores the sections which group game object attributes together."""

    ENDURANCE = {ComponentType.HEALTH, ComponentType.MOVEMENT_FORCE}
    DEFENCE = {ComponentType.ARMOUR, ComponentType.ARMOUR_REGEN_COOLDOWN}


class ComponentData(TypedDict, total=False):
    """Holds the data needed to initialise the components."""

    attributes: dict[ComponentType, tuple[int, int]]
    instant_effects: tuple[int, dict[ComponentType, Callable[[int], float]]]
    inventory_size: tuple[int, int]
    status_effects: tuple[
        int,
        dict[ComponentType, tuple[Callable[[int], float], Callable[[int], float]]],
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


# Record which component types are attack algorithms
ATTACK_ALGORITHMS = {
    ComponentType.AREA_OF_EFFECT_ATTACK,
    ComponentType.MELEE_ATTACK,
    ComponentType.RANGED_ATTACK,
}
