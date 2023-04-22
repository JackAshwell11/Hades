"""Tests all functions in game_objects/system.py."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING

# Custom
from hades.game_objects.base import GameObjectComponent
from hades.game_objects.system import ECS

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ()


class ComponentType(Enum):
    """Overrides the ComponentType enum to provide values for testing."""

    VALUE_ONE = auto()
    VALUE_TWO = auto()


class GameObjectComponentOne(GameObjectComponent):
    """Represents a valid game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.VALUE_ONE


class GameObjectComponentTwo(GameObjectComponent):
    """Represents a valid game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.VALUE_TWO


class GameObjectComponentEvent(GameObjectComponent):
    """Represents a valid game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.VALUE_ONE

    def __init__(self: GameObjectComponentEvent) -> None:
        """Initialise the object."""
        self.events: set[tuple[str, Callable]] = {("on_test_event", self.on_test_event)}

    @staticmethod
    def on_test_event(x: int) -> int:
        """Simulate an event for testing.

        Parameters
        ----------
        x: int
            A testing variable.
        """
        return x * 5


class GameObjectComponentInvalid:
    """Represents an invalid game object component useful for testing."""


def test_ecs_init() -> None:
    """Test that the entity component system is initialised correctly."""
    assert (
        repr(ECS()) == "<EntityComponentSystem (Game object count=0) (Event count=0)>"
    )


def test_ecs_zero_component_game_object() -> None:
    """Test the ECS with a game object that has no components."""


def test_ecs_multiple_component_game_object() -> None:
    """Test the ECS with a game object that has multiple components."""


def test_ecs_multiple_game_objects() -> None:
    """Test the ECS with multiple game objects."""


def test_ecs_event_game_object() -> None:
    """Test the ECS with a game object that has events."""


def test_ecs_invalid_component() -> None:
    """Test the ECS with a game object that has an invalid component."""


def test_dispatch_event_no_kwargs() -> None:
    """Test dispatching an event to the ECS with no keyword arguments."""


def test_ecs_dispatch_kwargs() -> None:
    """Test dispatching an event to the ECS with keyword arguments."""


def test_ecs_dispatch_to_multiple_handlers() -> None:
    """Test dispatching an event to the ECS that is received by multiple handlers."""


def test_ecs_dispatch_with_unregistered_event() -> None:
    """Test dispatching an unregistered event to the ECS."""


# def test_ecs_add_game_object_zero_components() -> None:
#     """Test that a game object is added with zero components."""
#
#
# def test_ecs_add_game_object_multiple_components() -> None:
#     """Test that a game object is added with one or more components."""
#     assert ecs.components == {
#         ComponentType.ACTIONABLE: {0},
#         ComponentType.COLLECTIBLE: {0},
#
#
# def test_ecs_add_game_object_invalid_component() -> None:
#     """Test that a game object is not added with an invalid component."""
#     with pytest.raises(expected_exception=AttributeError):
#
#
# def test_ecs_add_game_object_events() -> None:
#     """Test that a game object is added with a component that has events."""
#
#
# def test_ecs_remove_game_object_no_events() -> None:
#     """Test that a registered game object with no events is removed correctly."""
#
#
# def test_ecs_remove_game_object_no_components() -> None:
#     """Test that a registered game object with no components is removed correctly."""
#
#
# def test_ecs_remove_game_object_unregistered_object() -> None:
#     """Test that an unregistered game object is not removed."""
