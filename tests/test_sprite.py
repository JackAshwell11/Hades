# pylint: disable=redefined-outer-name
"""Tests all classes and functions in sprite.py."""

from __future__ import annotations

# Builtin
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

# Pip
import pytest

# Custom
from hades.constructors import GameObjectConstructor
from hades.sprite import AnimatedSprite, Bullet, HadesSprite
from hades_extensions.ecs import GameObjectType, Vec2d

if TYPE_CHECKING:
    from hades_extensions.ecs import Registry

__all__ = ()

# Create the texture path
texture_path = (
    Path(__file__).resolve().parent.parent / "src" / "hades" / "resources" / "textures"
)


@pytest.fixture
def constructor(request: pytest.FixtureRequest) -> GameObjectConstructor:
    """Create a game object constructor for testing.

    Args:
        request: The fixture request object.

    Returns:
        The game object constructor.
    """
    return GameObjectConstructor(
        "Test",
        "Test description",
        GameObjectType.Player,
        [f":resources:{param}" for param in request.param],
    )


@pytest.mark.parametrize(
    ("constructor", "position", "expected_result"),
    [
        (
            ["floor.png"],
            Vec2d(10, 20),
            [(672.0, 1312.0), texture_path / "floor.png"],
        ),
        (
            ["floor.png"],
            Vec2d(5, 10),
            [(352.0, 672.0), texture_path / "floor.png"],
        ),
        (
            ["floor.png"],
            Vec2d(0, 0),
            [(32.0, 32.0), texture_path / "floor.png"],
        ),
    ],
    indirect=["constructor"],
)
def test_hades_sprite_init(
    constructor: GameObjectConstructor,
    position: Vec2d,
    expected_result: tuple[tuple[float, float], Path],
) -> None:
    """Test that a HadesSprite object is initialised correctly.

    Args:
        constructor: The game object constructor for testing.
        position: The position of the sprite object.
        expected_result: The expected result of the test.
    """
    sprite = HadesSprite(
        Mock(),
        0,
        position,
        constructor,
    )
    assert sprite.position == expected_result[0]
    assert sprite.texture.file_path == expected_result[1]
    assert sprite.game_object_id == 0
    assert sprite.game_object_type == GameObjectType.Player
    assert sprite.name == "Test"
    assert sprite.description == "Test description"


def test_hades_sprite_update(registry: Registry) -> None:
    """Test that a HadesSprite object is updated correctly.

    Args:
        registry: The registry that manages the game objects, components, and systems.
    """
    # Create the sprite object and check that the position is correct
    constructor = GameObjectConstructor(
        "Test",
        "Test description",
        GameObjectType.Player,
        [":resources:floor.png"],
        kinematic=True,
    )
    sprite = HadesSprite(
        registry,
        registry.create_game_object(
            constructor.game_object_type,
            Vec2d(0, 0),
            constructor.components,
        ),
        Vec2d(0, 0),
        constructor,
    )
    assert sprite.position == (32.0, 32.0)

    # Set its position to (0, 0) and check that it is reset correctly
    sprite.position = (0, 0)
    sprite.update()
    assert sprite.position == (32.0, 32.0)


def test_bullet_init() -> None:
    """Test that a Bullet object is initialised correctly."""
    bullet = Bullet(Mock(), 0)
    assert bullet.game_object_id == 0
    assert bullet.game_object_type == GameObjectType.Bullet
    assert bullet.name == "Bullet"
    assert bullet.description == "A bullet that damages other game objects."


@pytest.mark.parametrize(
    ("constructor", "position", "expected_result"),
    [
        (
            ["floor.png"],
            Vec2d(10, 20),
            [(672.0, 1312.0), [texture_path / "floor.png"]],
        ),
        (
            ["floor.png"],
            Vec2d(5, 10),
            [(352.0, 672.0), [texture_path / "floor.png"]],
        ),
        (
            ["floor.png", "wall.png"],
            Vec2d(0, 0),
            [(32.0, 32.0), [texture_path / "floor.png", texture_path / "wall.png"]],
        ),
    ],
    indirect=["constructor"],
)
def test_animated_sprite_init(
    constructor: GameObjectConstructor,
    position: Vec2d,
    expected_result: tuple[tuple[float, float], list[Path]],
) -> None:
    """Test that an AnimatedSprite object is initialised correctly.

    Args:
        constructor: The game object constructor for testing.
        position: The position of the sprite object.
        expected_result: The expected result of the test.
    """
    sprite = AnimatedSprite(
        Mock(),
        0,
        position,
        constructor,
    )
    assert sprite.position == expected_result[0]
    assert sprite.texture.file_path == expected_result[1][0]
    assert sprite.game_object_id == 0
    assert sprite.game_object_type == GameObjectType.Player
    assert sprite.name == "Test"
    assert sprite.description == "Test description"
    assert len(sprite.sprite_textures) == len(expected_result[1])
    for i, textures in enumerate(sprite.sprite_textures):
        assert textures[0].file_path == expected_result[1][i]
        assert len(sprite.sprite_textures[0]) == 2


@pytest.mark.parametrize(
    ("constructor", "position", "expected_result"),
    [
        (
            [],
            Vec2d(10, 20),
            [IndexError, "list index out of range"],
        ),
        (
            ["floor.png"],
            Vec2d(-10, -20),
            [ValueError, "The position cannot be negative."],
        ),
    ],
    indirect=["constructor"],
)
def test_hades_sprite_errors(
    constructor: GameObjectConstructor,
    position: Vec2d,
    expected_result: tuple[type[Exception], str],
) -> None:
    """Test that a HadesSprite object raises the correct errors.

    Args:
        constructor: The game object constructor for testing.
        position: The position of the sprite object.
        expected_result: The expected result of the test.
    """
    with pytest.raises(
        expected_exception=expected_result[0],
        match=expected_result[1],
    ):
        HadesSprite(Mock(), 0, position, constructor)
