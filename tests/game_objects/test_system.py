"""Tests all functions in game_objects/system.py."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

# Pip
import pytest

# Custom
from hades.game_objects.base import ComponentType, GameObjectComponent
from hades.game_objects.system import ECS, ECSError

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData

__all__ = ()


class GameObjectComponentOne(GameObjectComponent):
    """Represents a game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.HEALTH


class GameObjectComponentTwo(GameObjectComponent):
    """Represents a game object component useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.ARMOUR


class GameObjectComponentData(GameObjectComponent):
    """Represents a game object component that has data useful for testing."""

    __slots__ = ("test_data",)

    # Class variables
    component_type: ComponentType = ComponentType.MONEY

    def __init__(
        self: GameObjectComponentData,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.test_data: int = component_data["test"]  # type: ignore[typeddict-item]


class GameObjectComponentInvalid:
    """Represents an invalid game object component useful for testing."""


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    return ECS()


def test_raise_not_registered_error() -> None:
    """Test that ECSError is raised correctly."""
    with pytest.raises(
        expected_exception=ECSError,
        match="The test `10` is not registered with the ECS.",
    ):
        raise ECSError(not_registered_type="test", value=10)


def test_raise_not_registered_error_custom_error() -> None:
    """Test that ECSError is raised correctly with a custom error message."""
    with pytest.raises(
        expected_exception=ECSError,
        match="The test `temp` error.",
    ):
        raise ECSError(
            not_registered_type="test",
            value="temp",
            error="error",
        )


def test_ecs_init(ecs: ECS) -> None:
    """Test that the entity component system is initialised correctly.

    Args:
        ecs: The entity component system for use in testing.
    """
    assert repr(ecs) == "<EntityComponentSystem (Game object count=0)>"


def test_ecs_game_object_with_zero_components(ecs: ECS) -> None:
    """Test the ECS with a game object that has no components.

    Args:
        ecs: The entity component system for use in testing.
    """
    # Test that adding the game object works correctly
    assert ecs.add_game_object({}) == 0
    assert ecs.get_components_for_game_object(0) == {}
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` does not have a steering object.",
    ):
        ecs.get_steering_object_for_game_object(0)

    # Test that removing the game object works correctly
    ecs.remove_game_object(0)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` is not registered with the ECS.",
    ):
        ecs.remove_game_object(0)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` is not registered with the ECS.",
    ):
        ecs.get_components_for_game_object(0)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` does not have a steering object.",
    ):
        ecs.get_steering_object_for_game_object(0)


def test_ecs_game_object_with_multiple_components(ecs: ECS) -> None:
    """Test the ECS with a game object that has multiple components.

    Args:
        ecs: The entity component system for use in testing.
    """
    # Test that adding the game object works correctly
    ecs.add_game_object({}, GameObjectComponentOne, GameObjectComponentTwo)
    assert list(ecs.get_components_for_game_object(0).keys()) == [
        ComponentType.HEALTH,
        ComponentType.ARMOUR,
    ]
    assert ecs.get_component_for_game_object(0, ComponentType.HEALTH)
    assert ecs.get_component_for_game_object(0, ComponentType.ARMOUR)
    assert len(ecs.get_components_for_component_type(ComponentType.HEALTH)) == 1
    assert len(ecs.get_components_for_component_type(ComponentType.ARMOUR)) == 1
    with pytest.raises(expected_exception=KeyError):
        ecs.get_component_for_game_object(0, ComponentType.MONEY)

    # Test that removing the game object works correctly
    ecs.remove_game_object(0)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` is not registered with the ECS.",
    ):
        ecs.get_components_for_game_object(0)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` is not registered with the ECS.",
    ):
        ecs.get_component_for_game_object(0, ComponentType.HEALTH)
    assert ecs.get_components_for_component_type(ComponentType.HEALTH) == []


def test_ecs_game_object_with_steering(ecs: ECS) -> None:
    """Test the ECS with a game object that has steering.

    Args:
        ecs: The entity component system for use in testing.
    """
    # Test that adding the game object with steering works correctly
    ecs.add_game_object({}, steering=True)
    steering_object = ecs.get_steering_object_for_game_object(0)
    assert steering_object.position == (0, 0)
    assert steering_object.velocity == (0, 0)

    # Test that removing the game object works correctly
    ecs.remove_game_object(0)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` does not have a steering object.",
    ):
        ecs.get_steering_object_for_game_object(0)


def test_ecs_multiple_game_objects(ecs: ECS) -> None:
    """Test the ECS with multiple game objects.

    Args:
        ecs: The entity component system for use in testing.
    """
    # Test that adding two game object works correctly
    assert ecs.add_game_object({}, GameObjectComponentOne) == 0
    assert ecs.add_game_object({}, GameObjectComponentOne, GameObjectComponentTwo) == 1
    assert ecs.get_component_for_game_object(0, ComponentType.HEALTH)
    assert ecs.get_component_for_game_object(1, ComponentType.HEALTH)
    assert ecs.get_component_for_game_object(1, ComponentType.ARMOUR)
    assert list(ecs.get_components_for_game_object(0).keys()) == [ComponentType.HEALTH]
    assert list(ecs.get_components_for_game_object(1).keys()) == [
        ComponentType.HEALTH,
        ComponentType.ARMOUR,
    ]

    # Test that removing the first game object works correctly
    ecs.remove_game_object(0)
    assert ecs.get_component_for_game_object(1, ComponentType.HEALTH)
    assert ecs.get_component_for_game_object(1, ComponentType.ARMOUR)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` is not registered with the ECS.",
    ):
        ecs.get_components_for_game_object(0)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` is not registered with the ECS.",
    ):
        ecs.get_component_for_game_object(0, ComponentType.HEALTH)


def test_ecs_component_data(ecs: ECS) -> None:
    """Test the ECS with a component that has data.

    Args:
        ecs: The entity component system for use in testing.
    """
    assert (
        ecs.add_game_object(
            {"test": 10},  # type: ignore[typeddict-unknown-key]
            GameObjectComponentData,
        )
        == 0
    )
    assert (
        cast(
            GameObjectComponentData,
            ecs.get_component_for_game_object(0, ComponentType.MONEY),
        ).test_data
        == 10
    )
    assert list(ecs.get_components_for_game_object(0).keys()) == [ComponentType.MONEY]


def test_ecs_nonexistent_component_data(ecs: ECS) -> None:
    """Test the ECS with a component that has data which is not provided.

    Args:
        ecs: The entity component system for use in testing.
    """
    with pytest.raises(expected_exception=KeyError):
        assert ecs.add_game_object({}, GameObjectComponentData) == 0


def test_ecs_duplicate_components(ecs: ECS) -> None:
    """Test the ECS with duplicate components for the same game object.

    Args:
        ecs: The entity component system for use in testing.
    """
    # Test that adding a game object with two of the same components raises an error
    with pytest.raises(
        expected_exception=ECSError,
        match=(
            "The component type `ComponentType.HEALTH` is already registered with the"
            " ECS."
        ),
    ):
        ecs.add_game_object({}, GameObjectComponentOne, GameObjectComponentOne)

    # Test that the game object does not exist
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` is not registered with the ECS.",
    ):
        ecs.get_component_for_game_object(0, ComponentType.HEALTH)


def test_ecs_duplicate_components_steering(ecs: ECS) -> None:
    """Test the ECS with duplicate components for the same game object with steering.

    Args:
        ecs: The entity component system for use in testing.
    """
    # Test that adding a game object with steering that has two of the same components
    # raises an error
    with pytest.raises(
        expected_exception=ECSError,
        match=(
            "The component type `ComponentType.HEALTH` is already registered with the"
            " ECS."
        ),
    ):
        ecs.add_game_object(
            {},
            GameObjectComponentOne,
            GameObjectComponentOne,
            steering=True,
        )

    # Test that the game object does not exist
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` is not registered with the ECS.",
    ):
        ecs.get_component_for_game_object(0, ComponentType.HEALTH)
    with pytest.raises(
        expected_exception=ECSError,
        match="The game object ID `0` does not have a steering object.",
    ):
        ecs.get_steering_object_for_game_object(0)


def test_ecs_invalid_component(ecs: ECS) -> None:
    """Test the ECS with a game object that has an invalid component.

    Args:
        ecs: The entity component system for use in testing.
    """
    with pytest.raises(expected_exception=AttributeError):
        ecs.add_game_object({}, GameObjectComponentInvalid)  # type: ignore[arg-type]
