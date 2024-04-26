# pylint: disable=redefined-outer-name
"""Tests all classes and functions in sprite.py."""

from __future__ import annotations

# Builtin
from pathlib import Path

# Pip
import pytest
from arcade import Texture, load_texture

# Custom
from hades.constants import GameObjectType
from hades.sprite import AnimatedSprite, HadesSprite
from hades_extensions.game_objects import Vec2d

__all__ = ()

# Create the texture path
texture_path = (
    Path(__file__).resolve().parent.parent / "src" / "hades" / "resources" / "textures"
)


@pytest.fixture()
def floor_texture() -> Texture:
    """Get a floor texture for testing.

    Returns:
        Texture: The floor texture for testing.
    """
    return load_texture(texture_path / "floor.png")


def test_hades_sprite_init(floor_texture: Texture) -> None:
    """Test that a HadesSprite object is initialised correctly.

    Args:
        floor_texture: The floor texture for testing.
    """
    sprite = HadesSprite((0, GameObjectType.PLAYER), Vec2d(10, 20), [floor_texture])
    assert sprite.position == (672.0, 1312.0)
    assert sprite.texture == floor_texture


def test_hades_sprite_negative_position(floor_texture: Texture) -> None:
    """Test that a HadesSprite object with a negative position raises an error.

    Args:
        floor_texture: The floor texture for testing.
    """
    with pytest.raises(
        expected_exception=ValueError,
        match="The position cannot be negative.",
    ):
        HadesSprite((0, GameObjectType.PLAYER), Vec2d(-10, -20), [floor_texture])


def test_hades_sprite_empty_textures() -> None:
    """Test that a HadesSprite object with no textures raises an error."""
    with pytest.raises(expected_exception=IndexError, match="list index out of range"):
        HadesSprite((0, GameObjectType.PLAYER), Vec2d(10, 20), [])


def test_hades_sprite_non_existent_texture() -> None:
    """Test that a HadesSprite object with a non-existent texture raises an error."""
    with pytest.raises(expected_exception=FileNotFoundError, match="non_existent.png"):
        HadesSprite(
            (0, GameObjectType.PLAYER),
            Vec2d(10, 20),
            [load_texture("non_existent.png")],
        )


def test_animated_sprite_init(floor_texture: Texture) -> None:
    """Test that an AnimatedSprite object is initialised correctly.

    Args:
        floor_texture: The floor texture for testing.
    """
    sprite = AnimatedSprite((0, GameObjectType.PLAYER), Vec2d(10, 20), [floor_texture])
    assert sprite.position == (672.0, 1312.0)
    assert sprite.texture == floor_texture
    assert len(sprite.sprite_textures) == 1
    assert len(sprite.sprite_textures[0]) == 2


def test_animated_sprite_multiple_textures(floor_texture: Texture) -> None:
    """Test that an AnimatedSprite object initialises with multiple textures.

    Args:
        floor_texture: The floor texture for testing.
    """
    sprite = AnimatedSprite(
        (1, GameObjectType.PLAYER),
        Vec2d(5, 10),
        [load_texture(texture_path / "wall.png"), floor_texture],
    )
    assert sprite.position == (352.0, 672.0)
    assert len(sprite.sprite_textures) == 2
    assert len(sprite.sprite_textures[0]) == 2
