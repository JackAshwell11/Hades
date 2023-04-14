"""Tests all functions in textures.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.exceptions import BiggerThanError
from hades.textures import (
    MovingTextureType,
    NonMovingTextureType,
    grid_pos_to_pixel,
    moving_textures,
    non_moving_textures,
)

__all__ = ()


def test_grid_pos_to_pixel_positive() -> None:
    """Test that a valid position is converted correctly."""
    assert grid_pos_to_pixel(500, 500) == (28028.0, 28028.0)


def test_grid_pos_to_pixel_zero() -> None:
    """Test that a zero position is converted correctly."""
    assert grid_pos_to_pixel(0, 0) == (28.0, 28.0)


def test_grid_pos_to_pixel_negative() -> None:
    """Test that a negative position is converted correctly."""
    with pytest.raises(expected_exception=BiggerThanError):
        grid_pos_to_pixel(-500, -500)


def test_grid_pos_to_pixel_string() -> None:
    """Test that a position made of strings is converted correctly."""
    with pytest.raises(expected_exception=TypeError):
        grid_pos_to_pixel("test", "test")  # type: ignore[arg-type]


def test_textures_non_moving() -> None:
    """Test the textures.py script produces a correct non-moving textures dict."""
    assert {
        key: value.name.split("\\")[-1] for key, value in non_moving_textures.items()
    } == {
        NonMovingTextureType.FLOOR: "floor.png-0-0-0-0-False-False-False-Simple ",
        NonMovingTextureType.WALL: "wall.png-0-0-0-0-False-False-False-Simple ",
        NonMovingTextureType.HEALTH_POTION: (
            "health_potion.png-0-0-0-0-False-False-False-Simple "
        ),
        NonMovingTextureType.ARMOUR_POTION: (
            "armour_potion.png-0-0-0-0-False-False-False-Simple "
        ),
        NonMovingTextureType.HEALTH_BOOST_POTION: (
            "health_boost_potion.png-0-0-0-0-False-False-False-Simple "
        ),
        NonMovingTextureType.ARMOUR_BOOST_POTION: (
            "armour_boost_potion.png-0-0-0-0-False-False-False-Simple "
        ),
        NonMovingTextureType.SPEED_BOOST_POTION: (
            "speed_boost_potion.png-0-0-0-0-False-False-False-Simple "
        ),
        NonMovingTextureType.FIRE_RATE_BOOST_POTION: (
            "fire_rate_boost_potion.png-0-0-0-0-False-False-False-Simple "
        ),
        NonMovingTextureType.SHOP: "shop.png-0-0-0-0-False-False-False-Simple ",
    }


def test_textures_moving() -> None:
    """Test the textures.py script produces a correct moving textures dict."""
    assert {
        key: [texture.name.split("\\")[-1] for texture in value]
        for key, value in moving_textures.items()
    } == {
        MovingTextureType.PLAYER_IDLE: [
            "player_idle.png-0-0-0-0-False-False-False-Simple ",
            "player_idle.png-0-0-0-0-True-False-False-Simple ",
        ],
        MovingTextureType.ENEMY_IDLE: [
            "enemy_idle.png-0-0-0-0-False-False-False-Simple ",
            "enemy_idle.png-0-0-0-0-True-False-False-Simple ",
        ],
    }
