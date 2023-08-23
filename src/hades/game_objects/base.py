"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from hades.game_objects.registry import Registry

__all__ = (
    "AttackAlgorithms",
    "GameObjectAttributeSectionType",
    "ComponentBase",
    "ComponentType",
    "SteeringBehaviours",
    "SteeringMovementState",
    "SystemBase",
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


@dataclass(slots=True)
class ComponentBase:
    """The base class for all components."""


class SystemBase:
    """The base class for all systems."""

    __slots__ = ("registry",)

    def __init__(self: SystemBase, registry: Registry) -> None:
        """Initialise the system.

        Args:
            registry: The registry that manages the game objects, components, and
            systems.
        """
        self.registry: Registry = registry

    def update(self: SystemBase, delta_time: float) -> None:
        """Process update logic for a system.

        Args:
            delta_time: The time interval since the last time the function was called.
        """

    def __repr__(self: SystemBase) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<{self.__class__.__name__} (Description=`{self.__doc__}`)>"
