# pylint: disable=redefined-outer-name
"""Tests all classes and functions in sprite.py."""

from __future__ import annotations

# Builtin
from pathlib import Path

# Pip
import pytest

# Custom
from hades.constructors import GameObjectConstructor, IconType
from hades.sprite import AnimatedSprite, HadesSprite, make_sprite
from hades_engine.ecs import GameObjectType

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
        "Test constructor",
        "Test description",
        GameObjectType.Player,
        0,
        request.param,
    )


@pytest.mark.parametrize(
    ("constructor", "expected_path"),
    [
        ([IconType.FLOOR], texture_path / "floor.png"),
    ],
    indirect=["constructor"],
)
def test_hades_sprite_init(
    constructor: GameObjectConstructor,
    expected_path: Path,
) -> None:
    """Test that a hades sprite object initialises correctly.

    Args:
        constructor: The game object constructor for testing.
        expected_path: The expected path of the texture.
    """
    sprite = HadesSprite(0, constructor)
    assert sprite.position == (0, 0)
    assert sprite.texture.file_path == expected_path
    assert sprite.game_object_id == 0
    assert sprite.game_object_type == GameObjectType.Player
    assert sprite.name == "Test constructor"
    assert sprite.description == "Test description"


def test_hades_sprite_init_no_texture() -> None:
    """Test that a hades sprite object raises an error when no textures are provided."""
    with pytest.raises(expected_exception=IndexError, match="list index out of range"):
        HadesSprite(
            0,
            GameObjectConstructor("Test", "Test", GameObjectType.Player, 0, []),
        )


@pytest.mark.parametrize(
    ("constructor", "expected_paths"),
    [
        ([IconType.FLOOR], [texture_path / "floor.png"]),
        (
            [IconType.FLOOR, IconType.WALL],
            [texture_path / "floor.png", texture_path / "wall.png"],
        ),
    ],
    indirect=["constructor"],
)
def test_animated_sprite_init(
    constructor: GameObjectConstructor,
    expected_paths: list[Path],
) -> None:
    """Test that an animated sprite object initialises correctly.

    Args:
        constructor: The game object constructor for testing.
        expected_paths: The expected path of the textures.
    """
    sprite = AnimatedSprite(0, constructor)
    assert sprite.position == (0, 0)
    assert sprite.texture.file_path == expected_paths[0]
    assert sprite.game_object_id == 0
    assert sprite.game_object_type == GameObjectType.Player
    assert sprite.name == "Test constructor"
    assert sprite.description == "Test description"
    assert len(sprite.sprite_textures) == len(expected_paths)
    for i, textures in enumerate(sprite.sprite_textures):
        assert textures[0].file_path == expected_paths[i]
        assert len(sprite.sprite_textures[0]) == 2


@pytest.mark.parametrize(
    ("constructor", "expected_sprite_type"),
    [
        ([IconType.FLOOR], HadesSprite),
        ([IconType.FLOOR, IconType.WALL], AnimatedSprite),
    ],
    indirect=["constructor"],
)
def test_make_sprite(
    constructor: GameObjectConstructor,
    expected_sprite_type: type[HadesSprite | AnimatedSprite],
) -> None:
    """Test that the make_sprite function creates the correct sprite object.

    Args:
        constructor: The game object constructor for testing.
        expected_sprite_type: The expected type of the sprite object.
    """
    sprite = make_sprite(0, constructor)
    assert isinstance(sprite, expected_sprite_type)
