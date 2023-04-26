"""Tests all functions in game_objects/system.py."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TypeVar

# Pip
import pytest

# Custom
from hades.game_objects.base import GameObjectComponent
from hades.game_objects.system import ECS, NotRegisteredError

__all__ = ()


# Define a generic type for the kwargs
T = TypeVar("T")


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


# TODO: CHANGE ALL GAME OBJECTS CREATIONS TO FIXTURES


class GameObjectComponentInvalid:
    """Represents an invalid game object component useful for testing."""


class GameObjectComponentEventOne(GameObjectComponent):
    """Represents a valid game object component with events useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.VALUE_ONE

    @staticmethod
    def event_test_no_kwarg() -> None:
        """Simulate a non-kwarg event for testing."""
        assert True

    @staticmethod
    def event_test_kwarg(x: str, **_: T) -> None:
        """Simulate a kwarg event for testing.

        Parameters
        ----------
        x: str
            A testing variable.
        """
        assert x == "test one"


class GameObjectComponentEventTwo(GameObjectComponent):
    """Represents a valid game object component with events useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.VALUE_TWO

    @staticmethod
    def event_test_kwarg(y: str, **_: T) -> None:
        """Simulate a kwarg event for testing.

        Parameters
        ----------
        y: str
            A testing variable.
        """
        assert y == "test two"

    @staticmethod
    def on_test_name() -> None:
        """Simulate an event that is not named properly for testing."""


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns
    -------
    ECS
        The entity component system for use in testing.
    """
    return ECS()


def test_ecs_init(ecs: ECS) -> None:
    """Test that the entity component system is initialised correctly.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    assert repr(ecs) == "<EntityComponentSystem (Game object count=0) (Event count=0)>"


def test_ecs_zero_component_game_object(ecs: ECS) -> None:
    """Test the ECS with a game object that has no components.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    # Test that adding the game object works correctly
    assert ecs.add_game_object() == 0

    # Test that removing the game object works correctly
    ecs.remove_game_object(0)
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The game object `0` is not registered with the ECS.",
    ):
        ecs.get_components_for_game_object(0)
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The game object `0` is not registered with the ECS.",
    ):
        ecs.remove_game_object(0)


def test_ecs_multiple_component_game_object(ecs: ECS) -> None:
    """Test the ECS with a game object that has multiple components.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    # Test that adding the game object works correctly
    component_one, component_two = GameObjectComponentOne(), GameObjectComponentTwo()
    ecs.add_game_object(component_one, component_two)
    assert ecs.get_components_for_game_object(0) == {component_one, component_two}
    assert ecs.get_game_objects_for_component_type(ComponentType.VALUE_ONE) == {0}
    assert ecs.get_game_objects_for_component_type(ComponentType.VALUE_TWO) == {0}

    # Test that removing the game object works correctly
    ecs.remove_game_object(0)
    assert ecs.get_game_objects_for_component_type(ComponentType.VALUE_ONE) == set()
    assert ecs.get_game_objects_for_component_type(ComponentType.VALUE_TWO) == set()


def test_ecs_multiple_game_objects(ecs: ECS) -> None:
    """Test the ECS with multiple game objects.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    # Test that adding two game object works correctly
    assert ecs.add_game_object() == 0
    assert ecs.add_game_object() == 1

    # Test that removing the first game object works correctly
    ecs.remove_game_object(0)
    assert ecs.get_components_for_game_object(1) == set()
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The game object `0` is not registered with the ECS.",
    ):
        ecs.get_components_for_game_object(0)


def test_ecs_event_game_object(ecs: ECS) -> None:
    """Test the ECS with a game object that has events.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    # Test that adding the game object works correctly
    component = GameObjectComponentEventOne()
    ecs.add_game_object(component)
    assert ecs.get_handlers_for_event_name("event_test_no_kwarg") == {
        component.event_test_no_kwarg,
    }

    # Test that removing the game object works correctly
    ecs.remove_game_object(0)
    assert ecs.get_handlers_for_event_name("event_test_no_kwarg") == set()


def test_ecs_bad_event_name(ecs: ECS) -> None:
    """Test that the ECS doesn't add events which aren't named properly.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    component = GameObjectComponentEventTwo()
    ecs.add_game_object(component)
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The event `on_test_name` is not registered with the ECS.",
    ):
        ecs.get_handlers_for_event_name("on_test_name")


def test_ecs_invalid_component(ecs: ECS) -> None:
    """Test the ECS with a game object that has an invalid component.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    with pytest.raises(expected_exception=AttributeError):
        ecs.add_game_object(GameObjectComponentInvalid())  # type: ignore[arg-type]


def test_ecs_unregistered_game_object_component_and_event(ecs: ECS) -> None:
    """Test that the ECS raises the correct errors for unregistered items.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    # Test that an unregistered game object raises an error
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The game object `0` is not registered with the ECS.",
    ):
        ecs.get_components_for_game_object(0)

    # Test that an unregistered component type raises an error
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The component type `VALUE_ONE` is not registered with the ECS.",
    ):
        ecs.get_game_objects_for_component_type(ComponentType.VALUE_ONE)

    # Test that an unregistered event raises an error
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The event `on_test` is not registered with the ECS.",
    ):
        ecs.get_handlers_for_event_name("on_test")


def test_dispatch_event_no_kwargs(ecs: ECS) -> None:
    """Test dispatching an event to the ECS with no keyword arguments.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    ecs.add_game_object(GameObjectComponentEventOne())
    ecs.dispatch_event("event_test_no_kwarg")


def test_ecs_dispatch_kwargs(ecs: ECS) -> None:
    """Test dispatching an event to the ECS with keyword arguments.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    ecs.add_game_object(GameObjectComponentEventTwo())
    ecs.dispatch_event("event_test_kwarg", y="test two")


def test_ecs_dispatch_to_multiple_handlers(ecs: ECS) -> None:
    """Test dispatching an event to the ECS that is received by multiple handlers.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    ecs.add_game_object(GameObjectComponentEventOne(), GameObjectComponentEventTwo())
    ecs.dispatch_event("event_test_kwarg", x="test one", y="test two")


def test_ecs_dispatch_with_unregistered_event(ecs: ECS) -> None:
    """Test dispatching an unregistered event to the ECS.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The event `on_test` is not registered with the ECS.",
    ):
        ecs.dispatch_event("on_test")
