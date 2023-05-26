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
    "GAME_OBJECT_ATTRIBUTES",
    "ATTACK_ALGORITHMS",
    "MOVEMENT_ALGORITHMS",
)


class ComponentType(Enum):
    """Stores the different types of components available."""

    AREA_OF_EFFECT_ATTACK = auto()
    ARMOUR = auto()
    ARMOUR_REGEN_COOLDOWN = auto()
    ATTACK_MANAGER = auto()
    FIRE_RATE_PENALTY = auto()
    HEALTH = auto()
    INSTANT_EFFECTS = auto()
    INVENTORY = auto()
    KEYBOARD_MOVEMENT = auto()
    MELEE_ATTACK = auto()
    MONEY = auto()
    MOVEMENT_FORCE = auto()
    MOVEMENT_MANAGER = auto()
    RANGED_ATTACK = auto()
    STATUS_EFFECTS = auto()
    STEERING_MOVEMENT = auto()
    VIEW_DISTANCE = auto()


class GameObjectAttributeSectionType(Enum):
    """Stores the sections which group game object attributes together."""

    ENDURANCE = {ComponentType.HEALTH, ComponentType.MOVEMENT_FORCE}
    DEFENCE = {ComponentType.ARMOUR, ComponentType.ARMOUR_REGEN_COOLDOWN}


class ComponentData(TypedDict, total=False):
    """Holds the data needed to initialise the components."""

    armour_regen: bool
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

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError


# Record which component types are game object attributes
GAME_OBJECT_ATTRIBUTES = {
    ComponentType.ARMOUR,
    ComponentType.ARMOUR_REGEN_COOLDOWN,
    ComponentType.FIRE_RATE_PENALTY,
    ComponentType.HEALTH,
    ComponentType.MOVEMENT_FORCE,
    ComponentType.VIEW_DISTANCE,
}

# Record which component types are attack algorithms
ATTACK_ALGORITHMS = {
    ComponentType.AREA_OF_EFFECT_ATTACK,
    ComponentType.MELEE_ATTACK,
    ComponentType.RANGED_ATTACK,
}

# Record which component types are movement algorithms
MOVEMENT_ALGORITHMS = {
    ComponentType.KEYBOARD_MOVEMENT,
    ComponentType.STEERING_MOVEMENT,
}
