"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple, TypeVar

if TYPE_CHECKING:
    from hades.game_objects.system import ECS

__all__ = (
    "ComponentType",
    "D",
    "EntityAttributeSectionType",
    "GameObjectComponent",
    "GameObjectConstructor",
)

# Define a type for the component_data
D = TypeVar("D")


class ComponentType(Enum):
    """Stores the different types of components available."""

    ARMOUR = auto()
    ARMOUR_REGEN_COOLDOWN = auto()
    INSTANT_EFFECT = auto()
    FIRE_RATE_PENALTY = auto()
    GRAPHICS = auto()
    HEALTH = auto()
    INVENTORY = auto()
    MONEY = auto()
    MOVEMENT_FORCE = auto()
    STATUS_EFFECT = auto()
    VIEW_DISTANCE = auto()


class EntityAttributeSectionType(Enum):
    """Stores the sections which group entity attributes together."""

    ENDURANCE = {ComponentType.HEALTH, ComponentType.MOVEMENT_FORCE}
    DEFENCE = {ComponentType.ARMOUR, ComponentType.ARMOUR_REGEN_COOLDOWN}


class GameObjectComponent:
    """The base class for all game object components."""

    # Class variables
    system: ECS
    component_type: ComponentType
    dependencies: set[ComponentType] = set()


class GameObjectConstructor(NamedTuple):
    """Represents a constructor for a game object with optional data."""

    components: tuple[type[GameObjectComponent], ...]
    component_data: dict[str, D]
