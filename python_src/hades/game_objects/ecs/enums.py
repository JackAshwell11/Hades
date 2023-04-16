"""Manages the different enums related to the entity component system."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hades.game_objects.ecs.system import System

__all__ = ("ComponentType", "GameObjectComponent")


class ComponentType(Enum):
    """Stores the different types of components available."""

    ACTIONABLE = auto()
    AREA_OF_EFFECT_ATTACK = auto()
    ARMOUR = auto()
    ARMOUR_REGEN = auto()
    ATTACKER = auto()
    COLLECTIBLE = auto()
    FIRE_RATE_PENALTY = auto()
    HEALTH = auto()
    INVENTORY = auto()
    MELEE_ATTACK = auto()
    MONEY = auto()
    RANGED_ATTACK = auto()
    SPEED_MULTIPLIER = auto()


class GameObjectComponent:
    """The base class for all game object components."""

    system: System

    def on_update(self: type[GameObjectComponent], *args, **kwargs) -> None:
        """Process the update event for the component."""
        raise NotImplementedError
