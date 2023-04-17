"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.system import EntityComponentSystem

__all__ = (
    "Actionable",
    "Collectible",
    "ComponentType",
    "GameObjectComponent",
    "ProcessorComponent",
)


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

    # Class variables
    system: EntityComponentSystem
    component_type: ComponentType


class ProcessorComponent(GameObjectComponent):
    """The base class for all processor components."""

    def process(self: type[GameObjectComponent], *args, **kwargs) -> None:
        """Process the update event for the processor."""
        raise NotImplementedError


class Actionable(GameObjectComponent):
    """Allows a game object to have an action when interacted with."""

    __slots__ = ("do_action",)

    # Class variables
    component_type: ComponentType = ComponentType.ACTIONABLE

    def __init__(self: Actionable, action_func: Callable[[None], bool]) -> None:
        """Initialise the object.

        Parameters
        ----------
        action_func: Callable[[None], bool]
            The callable which processes the action for this game object.
        """
        self.do_action: Callable[[None], bool] = action_func


class Collectible(GameObjectComponent):
    """Allows a game object to be collected when interacted with."""

    __slots__ = ("do_collect",)

    # Class variables
    component_type: ComponentType = ComponentType.COLLECTIBLE

    def __init__(self: Collectible, collect_func: Callable[[None], bool]) -> None:
        """Initialise the object.

        Parameters
        ----------
        collect_func: Callable[[None], bool]
            The callable which processes the collection for this game object.
        """
        self.do_collect: Callable[[None], bool] = collect_func
