"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
from enum import Flag, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.system import EntityComponentSystem

__all__ = (
    "Actionable",
    "Collectible",
    "ComponentType",
    "GameObjectComponent",
    "Processor",
)


class ComponentType(Flag):
    """Stores the different types of components available."""

    ACTIONABLE = auto()
    COLLECTIBLE = auto()
    INVENTORY = auto()
    GRAPHICS = auto()


class GameObjectComponent:
    """The base class for all game object components."""

    # Class variables
    system: EntityComponentSystem
    component_type: ComponentType


class Processor:
    """The base class for all processors."""

    # Class variables
    system: EntityComponentSystem
    entities: set[int]

    # TODO: USE PYGLET EVENTS TO MANAGE PROCESSOR UPDATES


class Actionable(GameObjectComponent):
    """Allows a game object to have an action when interacted with."""

    __slots__ = ("do_action",)

    # Class variables
    component_type: ComponentType = ComponentType.ACTIONABLE

    def __init__(self: Actionable, action_func: Callable[[], bool]) -> None:
        """Initialise the object.

        Parameters
        ----------
        action_func: Callable[[], bool]
            The callable which processes the action for this game object.
        """
        self.do_action: Callable[[], bool] = action_func


class Collectible(GameObjectComponent):
    """Allows a game object to be collected when interacted with."""

    __slots__ = ("do_collect",)

    # Class variables
    component_type: ComponentType = ComponentType.COLLECTIBLE

    def __init__(self: Collectible, collect_func: Callable[[], bool]) -> None:
        """Initialise the object.

        Parameters
        ----------
        collect_func: Callable[[], bool]
            The callable which processes the collection for this game object.
        """
        self.do_collect: Callable[[], bool] = collect_func
