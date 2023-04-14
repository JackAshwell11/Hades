"""Tests all functions in textures.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.textures import (
    grid_pos_to_pixel,
    moving_filenames,
    moving_textures,
    non_moving_filenames,
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
    with pytest.raises(expected_exception=ValueError):
        grid_pos_to_pixel(-500, -500)


def test_grid_pos_to_pixel_string() -> None:
    """Test that a position made of strings is converted correctly."""
    with pytest.raises(expected_exception=TypeError):
        grid_pos_to_pixel("test", "test")  # type: ignore[arg-type]


def test_textures_non_moving() -> None:
    """Test the textures.py script produces a correct non-moving textures dict."""
    # Compare the non_moving_texture dict to the non_moving_filenames dict
    for section_name, texture_list in non_moving_textures.items():
        for section_count, texture in enumerate(texture_list):
            assert non_moving_filenames[section_name][section_count] in texture.name


def test_textures_moving() -> None:
    """Test the textures.py script produces a correct moving textures dict."""
    # Compare the moving_filenames dict to the moving_textures dict
    for section_name, animations in moving_textures.items():
        for animation_type, texture_list in animations.items():
            for texture_count, textures in enumerate(texture_list):
                for flipped_texture in textures:
                    assert (
                        moving_filenames[section_name][animation_type][texture_count]
                        in flipped_texture.name
                    )
