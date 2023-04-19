"""Tests all functions in game_objects/system.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.exceptions import AlreadyAddedComponentError
from hades.game_objects.base import ComponentType, GameObjectComponent
from hades.game_objects.system import EntityComponentSystem

__all__ = ()


class GameObjectComponentOne(GameObjectComponent):
    """Represents a valid game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.ACTIONABLE


class GameObjectComponentTwo(GameObjectComponent):
    """Represents a valid game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.COLLECTIBLE


class InvalidGameObjectComponent:
    """Represents an invalid game object component useful for testing."""


def test_ecs_init() -> None:
    """Test that the entity component system is initialised correctly."""
    ecs = EntityComponentSystem()
    assert repr(ecs) == "<EntityComponentSystem (Game object count=0)>"
    assert ecs.game_objects == {}
    assert ecs.ids == {}


def test_ecs_add_game_object_zero_components() -> None:
    """Test that a game object is added with zero components."""
    ecs = EntityComponentSystem()
    assert ecs.add_game_object("temp") == 0
    assert ecs.game_objects == {0: {}}
    assert ecs.ids == {0: "temp"}


def test_ecs_add_game_object_multiple_components() -> None:
    """Test that a game object is added with one or more components."""
    ecs = EntityComponentSystem()
    component_one, component_two = (
        GameObjectComponentOne(),
        GameObjectComponentTwo(),
    )
    assert ecs.add_game_object("temp", component_one, component_two) == 0
    assert ecs.game_objects == {
        0: {
            ComponentType.ACTIONABLE: component_one,
            ComponentType.COLLECTIBLE: component_two,
        },
    }
    assert ecs.ids == {0: "temp"}


def test_ecs_remove_game_object_registered_object() -> None:
    """Test that a registered game object is correctly removed."""
    ecs = EntityComponentSystem()
    ecs.add_game_object("temp")
    ecs.remove_game_object(0)
    assert ecs.game_objects == {}
    assert ecs.ids == {}


def test_ecs_remove_game_object_unregistered_object() -> None:
    """Test that an unregistered game object is not removed."""
    ecs = EntityComponentSystem()
    with pytest.raises(expected_exception=KeyError):
        ecs.remove_game_object(0)


def test_ecs_add_component_to_game_object_registered_object() -> None:
    """Test that a component is added to a registered game object."""
    ecs, component = EntityComponentSystem(), GameObjectComponentOne()
    ecs.add_game_object("temp")
    ecs.add_component_to_game_object(0, component)
    assert ecs.game_objects == {0: {ComponentType.ACTIONABLE: component}}


def test_ecs_add_component_to_game_object_unregistered_object() -> None:
    """Test that a component is not added to an unregistered game object."""
    ecs, component = EntityComponentSystem(), GameObjectComponentOne()
    with pytest.raises(expected_exception=KeyError):
        ecs.add_component_to_game_object(0, component)


def test_ecs_add_component_to_game_object_invalid_component() -> None:
    """Test that an invalid component is not added to a registered game object."""
    ecs, component = EntityComponentSystem(), InvalidGameObjectComponent()
    ecs.add_game_object("temp")
    with pytest.raises(expected_exception=AttributeError):
        ecs.add_component_to_game_object(0, component)  # type: ignore[arg-type]


def test_ecs_add_component_to_game_object_registered_component() -> None:
    """Test that a registered component is not added to a registered game object."""
    ecs, component = EntityComponentSystem(), GameObjectComponentOne()
    ecs.add_game_object("temp", component)
    with pytest.raises(expected_exception=AlreadyAddedComponentError):
        ecs.add_component_to_game_object(0, component)


def test_ecs_remove_component_from_game_object_registered_object() -> None:
    """Test that a registered component is removed from a registered game object."""
    ecs, component = EntityComponentSystem(), GameObjectComponentOne()
    ecs.add_game_object("temp")
    ecs.add_component_to_game_object(0, component)
    ecs.remove_component_from_game_object(0, ComponentType.ACTIONABLE)
    assert ecs.game_objects == {0: {}}


def test_ecs_remove_component_from_game_object_unregistered_object() -> None:
    """Test if a component can't be removed from an unregistered game object."""
    ecs = EntityComponentSystem()
    with pytest.raises(expected_exception=KeyError):
        ecs.remove_component_from_game_object(0, ComponentType.ACTIONABLE)


def test_ecs_remove_component_from_game_object_unregistered_component() -> None:
    """Test if a component can't be removed from a registered game object."""
    ecs = EntityComponentSystem()
    ecs.add_game_object("temp")
    with pytest.raises(expected_exception=KeyError):
        ecs.remove_component_from_game_object(0, ComponentType.ACTIONABLE)
