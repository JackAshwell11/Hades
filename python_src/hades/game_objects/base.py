"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING, TypedDict

# Custom
from hades.textures import TextureType

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.system import ECS

__all__ = (
    "ComponentData",
    "ComponentType",
    "D",
    "EntityAttributeSectionType",
    "GameObjectComponent",
)

# Define a generic type for the kwargs parameter
D = int | str | bool | set[TextureType]


class ComponentType(Enum):
    """Stores the different types of components available."""

    ARMOUR = auto()
    ARMOUR_REGEN_COOLDOWN = auto()
    KEYBOARD_MOVEMENT = auto()
    INSTANT_EFFECTS = auto()
    FIRE_RATE_PENALTY = auto()
    HEALTH = auto()
    INVENTORY = auto()
    MONEY = auto()
    MOVEMENT_FORCE = auto()
    SPRITE = auto()
    STATUS_EFFECTS = auto()
    VIEW_DISTANCE = auto()


class EntityAttributeSectionType(Enum):
    """Stores the sections which group entity attributes together."""

    ENDURANCE = {ComponentType.HEALTH, ComponentType.MOVEMENT_FORCE}
    DEFENCE = {ComponentType.ARMOUR, ComponentType.ARMOUR_REGEN_COOLDOWN}


class ComponentData(TypedDict, total=False):
    """Holds the data needed to initialise the components."""

    attributes: dict[ComponentType, tuple[int, int]]
    blocking: bool
    inventory_height: int
    instant_effects: tuple[int, dict[ComponentType, Callable[[int], float]]]
    status_effects: tuple[
        int,
        dict[ComponentType, tuple[Callable[[int], float], Callable[[int], float]]],
    ]
    texture_types: set[TextureType]
    inventory_width: int


class GameObjectComponent:
    """The base class for all game object components."""

    # Class variables
    system: ECS
    component_type: ComponentType

    def __init__(self: GameObjectComponent, _: ComponentData) -> None:
        """Initialise the object."""
