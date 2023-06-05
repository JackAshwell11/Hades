"""Tests all functions in game_objects/movements.py."""
from __future__ import annotations

# Builtin
from typing import cast

# Pip
import pytest

# Custom
from hades.game_objects.attributes import MovementForce
from hades.game_objects.base import ComponentType
from hades.game_objects.movements import KeyboardMovement, SteeringMovement
from hades.game_objects.system import ECS

__all__ = ()


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    return ECS()


@pytest.fixture()
def keyboard_movement(ecs: ECS) -> KeyboardMovement:
    """Create a keyboard movement component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The keyboard movement component for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.MOVEMENT_FORCE: (100, 5)}},
        MovementForce,
        KeyboardMovement,
    )
    return cast(
        KeyboardMovement,
        ecs.get_component_for_game_object(0, ComponentType.MOVEMENTS),
    )


@pytest.fixture()
def steering_movement(ecs: ECS) -> SteeringMovement:
    """Create a steering movement component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The keyboard movement component for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.MOVEMENT_FORCE: (100, 5)}},
        MovementForce,
        SteeringMovement,
    )
    return cast(
        SteeringMovement,
        ecs.get_component_for_game_object(0, ComponentType.MOVEMENTS),
    )


def test_keyboard_movement_init(keyboard_movement: KeyboardMovement) -> None:
    """Test if the keyboard movement component is initialised correctly.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    assert (
        repr(keyboard_movement)
        == "<KeyboardMovement (North pressed=False) (South pressed=False) (East"
        " pressed=False) (West pressed=False)>"
    )


def test_keyboard_movement_calculate_force_none(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if no keys are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    assert keyboard_movement.calculate_force() == (0, 0)


def test_keyboard_movement_calculate_force_north(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated for a move north.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.north_pressed = True
    assert keyboard_movement.calculate_force() == (0, 100)


def test_keyboard_movement_calculate_force_south(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated for a move south.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.south_pressed = True
    assert keyboard_movement.calculate_force() == (0, -100)


def test_keyboard_movement_calculate_force_east(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated for a move east.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.east_pressed = True
    assert keyboard_movement.calculate_force() == (100, 0)


def test_keyboard_movement_calculate_force_west(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated for a move west.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.west_pressed = True
    assert keyboard_movement.calculate_force() == (-100, 0)


def test_keyboard_movement_calculate_force_east_west(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if east and west are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.east_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement.calculate_force() == (0, 0)


def test_keyboard_movement_calculate_force_north_south(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if north and south are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.east_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement.calculate_force() == (0, 0)


def test_keyboard_movement_calculate_force_north_west(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if north and west are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.north_pressed = True
    keyboard_movement.west_pressed = True
    assert keyboard_movement.calculate_force() == (-100, 100)


def test_keyboard_movement_calculate_force_south_east(
    keyboard_movement: KeyboardMovement,
) -> None:
    """Test if the correct force is calculated if south and east are pressed.

    Args:
        keyboard_movement: The keyboard movement component for use in testing.
    """
    keyboard_movement.south_pressed = True
    keyboard_movement.east_pressed = True
    assert keyboard_movement.calculate_force() == (100, -100)


def test_steering_movement_init(steering_movement: SteeringMovement) -> None:
    """Test if the steering movement component is initialised correctly.

    Args:
        steering_movement: The steering movement component for use in testing.
    """
    assert repr(steering_movement) == "<SteeringMovement>"
    # Temporary assert to increase coverage. Change this once enemy AI is done
    assert steering_movement.calculate_force() == (0, 0)
