"""Tests all classes and functions in sprite.py."""

from __future__ import annotations

# Pip
import pytest

# Custom
from hades.constants import GameObjectType
from hades.sprite import AnimatedSprite, BiggerThanError, HadesSprite, grid_pos_to_pixel

__all__ = ()


def test_raise_bigger_than_error() -> None:
    """Test that BiggerThanError is raised correctly."""
    with pytest.raises(
        expected_exception=BiggerThanError,
        match="The input must be bigger than or equal to 0.",
    ):
        raise BiggerThanError


def test_grid_pos_to_pixel_positive() -> None:
    """Test that a valid position is converted correctly."""
    assert grid_pos_to_pixel(500, 500) == (32032.0, 32032.0)


def test_grid_pos_to_pixel_zero() -> None:
    """Test that a zero position is converted correctly."""
    assert grid_pos_to_pixel(0, 0) == (32.0, 32.0)


def test_grid_pos_to_pixel_x_negative() -> None:
    """Test that a negative x position raises an error."""
    with pytest.raises(
        expected_exception=BiggerThanError,
        match="The input must be bigger than or equal to 0.",
    ):
        grid_pos_to_pixel(-500, 500)


def test_grid_pos_to_pixel_y_negative() -> None:
    """Test that a negative y position raises an error."""
    with pytest.raises(
        expected_exception=BiggerThanError,
        match="The input must be bigger than or equal to 0.",
    ):
        grid_pos_to_pixel(500, -500)


def test_grid_pos_to_pixel_both_negative() -> None:
    """Test that a negative x and y position raises an error."""
    with pytest.raises(
        expected_exception=BiggerThanError,
        match="The input must be bigger than or equal to 0.",
    ):
        grid_pos_to_pixel(-500, -500)


def test_grid_pos_to_pixel_string() -> None:
    """Test that a position made of strings is converted correctly."""
    with pytest.raises(expected_exception=TypeError):
        grid_pos_to_pixel("test", "test")  # type: ignore[arg-type]


def test_hades_sprite_init() -> None:
    """Test that a HadesSprite object is initialised correctly."""
    sprite = HadesSprite((0, GameObjectType.PLAYER), (10, 20), ["floor.png"])
    assert sprite.position == (672.0, 1312.0)
    assert (
        repr(sprite)
        == "<HadesSprite (Game object ID=0) (Current texture=<Texture"
        " cache_name=b3d8c789f0ab79a64f6ee6c8eac8fc329b53a3a56ed6c0ee262522cefef5dcf4|("
        "0, 1, 2, 3)|SimpleHitBoxAlgorithm|>)>"
    )


def test_hades_sprite_negative_position() -> None:
    """Test that a HadesSprite object with a negative position raises an error."""
    with pytest.raises(
        expected_exception=BiggerThanError,
        match="The input must be bigger than or equal to 0.",
    ):
        HadesSprite((0, GameObjectType.PLAYER), (-10, -20), ["floor.png"])


def test_hades_sprite_empty_textures() -> None:
    """Test that a HadesSprite object with no textures raises an error."""
    with pytest.raises(expected_exception=IndexError):
        HadesSprite((0, GameObjectType.PLAYER), (10, 20), [])


def test_hades_sprite_non_existent_texture() -> None:
    """Test that a HadesSprite object with a non-existent texture raises an error."""
    with pytest.raises(expected_exception=FileNotFoundError):
        HadesSprite((0, GameObjectType.PLAYER), (10, 20), ["test.png"])


def test_animated_sprite_init() -> None:
    """Test that an AnimatedSprite object is initialised correctly."""
    sprite = AnimatedSprite((0, GameObjectType.PLAYER), (10, 20), ["floor.png"])
    assert sprite.position == (672.0, 1312.0)
    assert len(sprite.sprite_textures) == 1
    assert len(sprite.sprite_textures[0]) == 2
    assert (
        repr(sprite)
        == "<AnimatedSprite (Game object ID=0) (Current texture=<Texture"
        " cache_name=b3d8c789f0ab79a64f6ee6c8eac8fc329b53a3a56ed6c0ee262522cefef5dcf4|("
        "0, 1, 2, 3)|SimpleHitBoxAlgorithm|>)>"
    )


def test_animated_sprite_multiple_textures() -> None:
    """Test that an AnimatedSprite object initialises with multiple textures."""
    sprite = AnimatedSprite(
        (1, GameObjectType.PLAYER),
        (5, 10),
        ["wall.png", "floor.png"],
    )
    assert sprite.position == (352.0, 672.0)
    assert len(sprite.sprite_textures) == 2
    assert len(sprite.sprite_textures[0]) == 2
    assert (
        repr(sprite)
        == "<AnimatedSprite (Game object ID=1) (Current texture=<Texture"
        " cache_name=6c519a52622fb21c14df6fddfe3541a38344e258ade26b2f02505bb216f73b32|("
        "0, 1, 2, 3)|SimpleHitBoxAlgorithm|>)>"
    )
