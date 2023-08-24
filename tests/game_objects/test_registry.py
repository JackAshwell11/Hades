"""Tests all classes and functions in game_objects/registry.py."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass

# Pip
import pytest

# Custom
from hades.game_objects.base import ComponentBase, SystemBase
from hades.game_objects.registry import Registry, RegistryError
from hades.game_objects.steering import Vec2d

__all__ = ()


@dataclass(slots=True)
class GameObjectComponentOne(ComponentBase):
    """Represents a game object component useful for testing."""


@dataclass(slots=True)
class GameObjectComponentTwo(ComponentBase):
    """Represents a game object component useful for testing."""

    test_list: list[int]
    test_int: int = 0


class GameObjectComponentInvalid:
    """Represents an invalid game object component useful for testing."""


class SystemTest(SystemBase):
    """Represents a test system useful for testing."""

    called: bool = False

    def update(self: SystemTest, _: float) -> None:
        """Update the system."""
        self.called = True


@pytest.fixture()
def registry() -> Registry:
    """Create a registry for use in testing.

    Returns:
        The registry for use in testing.
    """
    return Registry()


def test_raise_not_registered_error() -> None:
    """Test that RegistryError is raised correctly."""
    with pytest.raises(
        expected_exception=RegistryError,
        match="The test `10` is not registered with the registry.",
    ):
        raise RegistryError(not_registered_type="test", value=10)


def test_raise_not_registered_error_custom_error() -> None:
    """Test that RegistryError is raised correctly with a custom error message."""
    with pytest.raises(
        expected_exception=RegistryError,
        match="The test `10` error.",
    ):
        raise RegistryError(
            not_registered_type="test",
            value=10,
            error="error",
        )


def test_registry_init(registry: Registry) -> None:
    """Test that the registry is initialised correctly.

    Args:
        registry: The registry for use in testing.
    """
    assert (
        repr(registry)
        == "<Registry (Game object count=0) (Component count=0) (System count=0)>"
    )


def test_registry_game_object_with_zero_components(registry: Registry) -> None:
    """Test the registry with a game object that has no components.

    Args:
        registry: The registry for use in testing.
    """
    # Test that adding the game object works correctly
    assert registry.create_game_object() == 0
    assert not list(registry.get_components(GameObjectComponentOne))
    assert not list(registry.get_components(GameObjectComponentTwo))
    assert registry.walls == set()
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `0` does not have a kinematic object.",
    ):
        registry.get_kinematic_object_for_game_object(0)
    with pytest.raises(expected_exception=KeyError):
        registry.get_component_for_game_object(0, GameObjectComponentOne)
    with pytest.raises(expected_exception=KeyError):
        registry.get_component_for_game_object(0, GameObjectComponentTwo)

    # Test that removing the game object works correctly
    registry.delete_game_object(0)
    assert not list(registry.get_components(GameObjectComponentOne))
    assert not list(registry.get_components(GameObjectComponentTwo))
    assert registry.walls == set()
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `0` is not registered with the registry.",
    ):
        registry.delete_game_object(0)


def test_registry_game_object_with_multiple_components(registry: Registry) -> None:
    """Test the registry with a game object that has multiple components.

    Args:
        registry: The registry for use in testing.
    """
    # Test that adding the game object works correctly
    registry.create_game_object(GameObjectComponentOne(), GameObjectComponentTwo([10]))
    assert list(registry.get_components(GameObjectComponentOne)) == [
        (0, (GameObjectComponentOne(),)),
    ]
    assert list(registry.get_components(GameObjectComponentTwo)) == [
        (0, (GameObjectComponentTwo([10]),)),
    ]
    assert list(
        registry.get_components(GameObjectComponentOne, GameObjectComponentTwo),
    ) == [
        (0, (GameObjectComponentOne(), GameObjectComponentTwo([10]))),
    ]
    assert (
        registry.get_component_for_game_object(0, GameObjectComponentOne)
        == GameObjectComponentOne()
    )
    assert registry.get_component_for_game_object(
        0,
        GameObjectComponentTwo,
    ) == GameObjectComponentTwo([10])
    with pytest.raises(expected_exception=KeyError):
        registry.get_component_for_game_object(
            0,
            GameObjectComponentInvalid,  # type: ignore[type-var]
        )

    # Test that removing the game object works correctly
    registry.delete_game_object(0)
    assert not list(
        registry.get_components(GameObjectComponentOne, GameObjectComponentTwo),
    )
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `0` is not registered with the registry.",
    ):
        registry.get_component_for_game_object(0, GameObjectComponentOne)


def test_registry_game_object_with_steering(registry: Registry) -> None:
    """Test the registry with a game object that has steering.

    Args:
        registry: The registry for use in testing.
    """
    # Test that adding the game object with steering works correctly
    registry.create_game_object(kinematic=True)
    kinematic_object = registry.get_kinematic_object_for_game_object(0)
    assert kinematic_object.position == Vec2d(0, 0)
    assert kinematic_object.rotation == 0
    assert kinematic_object.velocity == Vec2d(0, 0)

    # Test that removing the game object works correctly
    registry.delete_game_object(0)
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `0` does not have a kinematic object.",
    ):
        registry.get_kinematic_object_for_game_object(0)


def test_registry_multiple_game_objects(registry: Registry) -> None:
    """Test the registry with multiple game objects.

    Args:
        registry: The registry for use in testing.
    """
    # Test that adding two game object works correctly
    assert registry.create_game_object(GameObjectComponentOne()) == 0
    assert (
        registry.create_game_object(
            GameObjectComponentOne(),
            GameObjectComponentTwo([10]),
        )
        == 1
    )
    assert list(registry.get_components(GameObjectComponentOne)) == [
        (0, (GameObjectComponentOne(),)),
        (1, (GameObjectComponentOne(),)),
    ]
    assert list(registry.get_components(GameObjectComponentTwo)) == [
        (1, (GameObjectComponentTwo([10]),)),
    ]
    assert (
        registry.get_component_for_game_object(0, GameObjectComponentOne)
        == GameObjectComponentOne()
    )
    assert (
        registry.get_component_for_game_object(1, GameObjectComponentOne)
        == GameObjectComponentOne()
    )
    assert registry.get_component_for_game_object(
        1,
        GameObjectComponentTwo,
    ) == GameObjectComponentTwo([10])

    # Test that removing the first game object works correctly
    registry.delete_game_object(0)
    assert list(registry.get_components(GameObjectComponentOne)) == [
        (1, (GameObjectComponentOne(),)),
    ]
    assert list(registry.get_components(GameObjectComponentTwo)) == [
        (1, (GameObjectComponentTwo([10]),)),
    ]
    assert (
        registry.get_component_for_game_object(1, GameObjectComponentOne)
        == GameObjectComponentOne()
    )
    assert registry.get_component_for_game_object(
        1,
        GameObjectComponentTwo,
    ) == GameObjectComponentTwo([10])
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `0` is not registered with the registry.",
    ):
        registry.get_component_for_game_object(0, GameObjectComponentOne)


def test_registry_duplicate_components(registry: Registry) -> None:
    """Test the registry with duplicate components for the same game object.

    Args:
        registry: The registry for use in testing.
    """
    # Test that adding a game object with two of the same components only adds the first
    # one
    registry.create_game_object(
        GameObjectComponentTwo([10]),
        GameObjectComponentTwo([20]),
    )
    assert list(registry.get_components(GameObjectComponentTwo)) == [
        (0, (GameObjectComponentTwo([10]),)),
    ]
    assert registry.get_component_for_game_object(
        0,
        GameObjectComponentTwo,
    ) == GameObjectComponentTwo([10])


def test_registry_zero_systems(registry: Registry) -> None:
    """Test the registry with registering zero systems.

    Args:
        registry: The registry for use in testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match=(
            "The system `<class 'tests.game_objects.test_registry.SystemTest'>` is not "
            "registered with the registry."
        ),
    ):
        registry.get_system(SystemTest)


def test_registry_valid_system(registry: Registry) -> None:
    """Test the registry with a valid system.

    Args:
        registry: The registry for use in testing.
    """
    # Test that registering a system works correctly
    system = SystemTest(registry)
    registry.add_system(system)
    with pytest.raises(
        expected_exception=RegistryError,
        match="The system `SystemTest` is already registered with the registry.",
    ):
        registry.add_system(system)
    assert registry.get_system(SystemTest) == system
    registry.update(0)
    assert system.called


def test_registry_add_wall(registry: Registry) -> None:
    """Test the registry with adding a wall.

    Args:
        registry: The registry for use in testing.
    """
    assert registry.walls == set()
    registry.walls.add(Vec2d(0, 0))
    registry.walls.add(Vec2d(1, 1))
    registry.walls.add(Vec2d(0, 0))
    assert registry.walls == {Vec2d(0, 0), Vec2d(1, 1)}


def test_registry_component_copy(registry: Registry) -> None:
    """Test that the registry copies components correctly when creating a game object.

    Args:
        registry: The registry for use in testing.
    """
    # Create the game objects
    game_object_component_two = GameObjectComponentTwo([])
    registry.create_game_object(game_object_component_two)
    registry.create_game_object(game_object_component_two)

    # Modify the components
    game_object_component_two.test_list.append(10)
    game_object_component_two.test_int = 10
    game_object_component_two_zero = registry.get_component_for_game_object(
        0,
        GameObjectComponentTwo,
    )
    game_object_component_two_zero.test_list.append(20)
    game_object_component_two_zero.test_int = 20
    game_object_component_two_one = registry.get_component_for_game_object(
        1,
        GameObjectComponentTwo,
    )
    game_object_component_two_one.test_list.append(30)
    game_object_component_two_one.test_int = 30

    # Check that the components are different
    assert game_object_component_two.test_list == [10]
    assert game_object_component_two.test_int == 10
    assert game_object_component_two_zero.test_list == [20]
    assert game_object_component_two_zero.test_int == 20
    assert game_object_component_two_one.test_list == [30]
    assert game_object_component_two_one.test_int == 30
