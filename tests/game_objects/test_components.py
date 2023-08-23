"""Tests all classes and functions in game_objects/components.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.game_objects.components import GameObjectAttributeBase

__all__ = ()


class TestGameObjectAttribute(GameObjectAttributeBase):
    """Represents a game object attribute useful for testing."""


@pytest.fixture()
def game_object_attribute() -> TestGameObjectAttribute:
    """Create a game object attribute for use in testing.

    Returns:
        The game object attribute for use in testing.
    """
    return TestGameObjectAttribute(150, 3)


def test_game_object_attribute_setter_higher(
    game_object_attribute: TestGameObjectAttribute,
) -> None:
    """Test that a game object attribute is set with a higher value correctly.

    Args:
        game_object_attribute: The game object attribute for use in testing.
    """
    game_object_attribute.value = 200
    assert game_object_attribute.value == 150


def test_game_object_attribute_setter_lower(
    game_object_attribute: TestGameObjectAttribute,
) -> None:
    """Test that a game object attribute is set with a lower value correctly.

    Args:
        game_object_attribute: The game object attribute for use in testing.
    """
    game_object_attribute.value = 100
    assert game_object_attribute.value == 100


def test_game_object_attribute_setter_iadd(
    game_object_attribute: TestGameObjectAttribute,
) -> None:
    """Test that adding a value to the game object attribute is correct.

    Args:
        game_object_attribute: The game object attribute for use in testing.
    """
    game_object_attribute.value += 100
    assert game_object_attribute.value == 150


def test_game_object_attribute_setter_isub(
    game_object_attribute: TestGameObjectAttribute,
) -> None:
    """Test that subtracting a value from the game object attribute is correct.

    Args:
        game_object_attribute: The game object attribute for use in testing.
    """
    game_object_attribute.value -= 200
    assert game_object_attribute.value == 0
