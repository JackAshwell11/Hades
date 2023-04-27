"""Tests all functions in game_objects/system.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.game_objects.base import ComponentType, GameObjectComponent
from hades.game_objects.system import ECS, NotRegisteredError

__all__ = ()


class GameObjectComponentOne(GameObjectComponent):
    """Represents a valid game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.HEALTH


class GameObjectComponentTwo(GameObjectComponent):
    """Represents a valid game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.ARMOUR


class GameObjectComponentInvalid:
    """Represents an invalid game object component useful for testing."""


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns
    -------
    ECS
        The entity component system for use in testing.
    """
    return ECS()


@pytest.fixture()
def game_object_component_one() -> GameObjectComponentOne:
    """Create a valid game object component for use in testing.

    Returns
    -------
    GameObjectComponentOne
        The valid game object component for use in testing.
    """
    return GameObjectComponentOne()


@pytest.fixture()
def game_object_component_two() -> GameObjectComponentTwo:
    """Create a valid game object component for use in testing.

    Returns
    -------
    GameObjectComponentTwo
        The valid game object component for use in testing.
    """
    return GameObjectComponentTwo()


@pytest.fixture()
def game_object_component_invalid() -> GameObjectComponentInvalid:
    """Create an invalid game object component for use in testing.

    Returns
    -------
    GameObjectComponentInvalid
        The invalid game object component for use in testing.
    """
    return GameObjectComponentInvalid()


def test_raise_not_registered_error() -> None:
    """Test that NotRegisteredError is raised correctly."""
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The test `10` is not registered with the ECS.",
    ):
        raise NotRegisteredError(not_registered_type="test", value=10)


def test_ecs_init(ecs: ECS) -> None:
    """Test that the entity component system is initialised correctly.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    """
    assert repr(ecs) == "<EntityComponentSystem (Game object count=0)>"


def test_ecs_game_object_with_zero_components(ecs: ECS) -> None:
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


def test_ecs_game_object_with_multiple_components(
    ecs: ECS,
    game_object_component_one: GameObjectComponentOne,
    game_object_component_two: GameObjectComponentTwo,
) -> None:
    """Test the ECS with a game object that has multiple components.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    game_object_component_one: GameObjectComponentOne
        The first game object component for use in testing.
    game_object_component_two: GameObjectComponentTwo
        The second game object components for use in testing.
    """
    # Test that adding the game object works correctly
    ecs.add_game_object(game_object_component_one, game_object_component_two)
    assert ecs.get_components_for_game_object(0) == {
        ComponentType.HEALTH: game_object_component_one,
        ComponentType.ARMOUR: game_object_component_two,
    }

    # Test that removing the game object works correctly
    ecs.remove_game_object(0)
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The game object `0` is not registered with the ECS.",
    ):
        ecs.get_components_for_game_object(0)


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
    assert ecs.get_components_for_game_object(1) == {}
    with pytest.raises(
        expected_exception=NotRegisteredError,
        match="The game object `0` is not registered with the ECS.",
    ):
        ecs.get_components_for_game_object(0)


def test_ecs_invalid_component(
    ecs: ECS,
    game_object_component_invalid: GameObjectComponentInvalid,
) -> None:
    """Test the ECS with a game object that has an invalid component.

    Parameters
    ----------
    ecs: ECS
        The entity component system for use in testing.
    game_object_component_invalid: GameObjectComponentInvalid
        The invalid game object component for use in testing.
    """
    with pytest.raises(expected_exception=AttributeError):
        ecs.add_game_object(game_object_component_invalid)  # type: ignore[arg-type]
