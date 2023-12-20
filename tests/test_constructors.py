"""Tests all classes and functions in constructors.py."""

from __future__ import annotations

# Pip
import pytest

# Custom
from hades.constructors import GameObjectConstructorManager, GameObjectType

__all__ = ()


@pytest.mark.parametrize("game_object_type", list(GameObjectType))
def test_game_object_constructor_manager(game_object_type: GameObjectType) -> None:
    """Test that retrieving a constructor works correctly for all game object types."""
    assert GameObjectConstructorManager.get_constructor(game_object_type)


def test_game_object_constructor_manager_non_existent_type() -> None:
    """Test that retrieving a non-existent constructor raises a KeyError."""
    with pytest.raises(KeyError):
        GameObjectConstructorManager.get_constructor("TEST")  # type: ignore[arg-type]
