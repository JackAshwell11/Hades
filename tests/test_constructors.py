"""Tests all classes and functions in constructors.py."""

from __future__ import annotations

# Pip
import pytest

# Custom
from hades.constructors import (
    GameObjectConstructor,
    IconType,
    game_object_constructors,
    texture_cache,
)
from hades_extensions.ecs import GameObjectType


@pytest.mark.parametrize(
    ("game_object_type", "expected_result"),
    [
        (GameObjectType.Wall, ("Wall", "A wall that blocks movement.")),
        (GameObjectType.Floor, ("Floor", "A floor that allows movement.")),
        (GameObjectType.Player, ("Player", "The player character.")),
        (GameObjectType.Enemy, ("Enemy", "An enemy character.")),
        (GameObjectType.Goal, ("Goal", "The goal of the level.")),
        (
            GameObjectType.HealthPotion,
            ("Health Potion", "A potion that restores health."),
        ),
        (GameObjectType.Chest, ("Chest", "A chest that contains loot.")),
        (
            GameObjectType.Bullet,
            ("Bullet", "A bullet that damages other game objects."),
        ),
    ],
)
def test_factory_functions(
    game_object_type: GameObjectType,
    expected_result: tuple[str, str],
) -> None:
    """Test that the factory functions return the expected results.

    Args:
        game_object_type: The game object type to test.
        expected_result: The expected results.
    """
    constructor = game_object_constructors[game_object_type]
    assert isinstance(constructor, GameObjectConstructor)
    assert constructor.game_object_type == game_object_type
    assert constructor.name == expected_result[0]
    assert constructor.description == expected_result[1]
    for icon_type in constructor.textures:
        assert texture_cache[f":resources:textures/{icon_type.value}"] is not None


def test_duplicate_texture_paths() -> None:
    """Test that only one texture is loaded for duplicate texture paths."""
    constructor = GameObjectConstructor(
        "Test duplicate",
        "Test description",
        GameObjectType.Player,
        0,
        [IconType.FLOOR, IconType.FLOOR],
    )
    assert len(constructor.textures) == 2
    assert texture_cache[":resources:textures/floor.png"] is not None
