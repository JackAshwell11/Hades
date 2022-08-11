"""Tests all functions in textures.py."""
from __future__ import annotations

# Pip
import arcade
import pytest

# Custom
from game.textures import (
    grid_pos_to_pixel,
    moving_filenames,
    moving_textures,
    non_moving_filenames,
    non_moving_textures,
)

__all__ = ()


def test_grid_pos_to_pixel() -> None:
    """Test the grid_pos_to_pixel function in textures.py."""
    assert grid_pos_to_pixel(500, 500) == (28028.0, 28028.0)
    assert grid_pos_to_pixel(0, 0) == (28.0, 28.0)
    with pytest.raises(ValueError):
        grid_pos_to_pixel(-500, -500)
    with pytest.raises(TypeError):
        grid_pos_to_pixel("test", "test")  # type: ignore


def test_textures_script() -> None:
    """Test the textures.py script."""
    # Compare the non_moving_filenames dict to the non_moving_textures dict
    for section_name, non_moving_type in non_moving_filenames.items():
        for section_count, texture_filename in enumerate(non_moving_type):
            compare_texture = non_moving_textures[section_name][section_count]
            assert (
                isinstance(compare_texture, arcade.Texture)
                and texture_filename in compare_texture.name
            )

    # Compare the moving_filenames dict to the moving_textures dict
    for section_name, moving_type in moving_filenames.items():
        for animation_type, filenames in moving_type.items():
            for texture_count, texture_filename in enumerate(filenames):
                compare_texture = moving_textures[section_name][animation_type][
                    texture_count
                ]
                for texture in compare_texture:
                    assert (
                        isinstance(texture, arcade.Texture)
                        and texture_filename in texture.name
                    )
