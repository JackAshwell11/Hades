"""Tests all classes and functions in constructors.py."""

from __future__ import annotations

# Pip
import pytest

# Custom
from hades.constructors import (
    GameObjectConstructor,
    game_object_constructors,
    texture_cache,
)
from hades_extensions.game_objects import GameObjectType
from hades_extensions.game_objects.components import KinematicComponent


@pytest.fixture(autouse=True)
def _clear_texture_cache() -> None:
    """Clear the texture cache before each test."""
    texture_cache.clear()


@pytest.mark.parametrize(
    ("game_object_type", "expected_result"),
    [
        (GameObjectType.Wall, ("Wall", "A wall that blocks movement.", True, False)),
        (
            GameObjectType.Floor,
            ("Floor", "A floor that allows movement.", False, False),
        ),
        (GameObjectType.Player, ("Player", "The player character.", False, True)),
        (GameObjectType.Enemy, ("Enemy", "An enemy character.", False, True)),
        (
            GameObjectType.Potion,
            ("Health Potion", "A potion that restores health.", False, False),
        ),
        (
            GameObjectType.Bullet,
            ("Bullet", "A bullet that damages other game objects.", False, False),
        ),
    ],
)
def test_factory_functions(
    game_object_type: GameObjectType,
    expected_result: tuple[str, str, bool, bool],
) -> None:
    """Test that the factory functions return the expected results.

    Args:
        game_object_type: The game object type to test.
        expected_result: The expected results.
    """
    constructor = game_object_constructors[game_object_type]()
    assert isinstance(constructor, GameObjectConstructor)
    assert constructor.game_object_type == game_object_type
    assert constructor.name == expected_result[0]
    assert constructor.description == expected_result[1]
    assert constructor.static == expected_result[2]
    assert constructor.kinematic == expected_result[3]
    assert (
        KinematicComponent in [type(component) for component in constructor.components]
    ) == (constructor.static or constructor.kinematic)
    for texture_path in constructor.texture_paths:
        assert texture_cache[texture_path] is not None


def test_nonexistent_texture_path() -> None:
    """Test that a FileNotFoundError is raised when a texture path does not exist."""
    with pytest.raises(expected_exception=FileNotFoundError, match="non_existent.png"):
        GameObjectConstructor(
            "Test",
            "Test description",
            GameObjectType.Player,
            ["non_existent.png"],
            kinematic=True,
        )
