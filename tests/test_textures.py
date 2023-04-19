"""Tests all functions in textures.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.exceptions import BiggerThanError
from hades.textures import (
    grid_pos_to_pixel,
    load_moving_texture,
    load_non_moving_texture,
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


def test_load_moving_texture_valid_filename() -> None:
    """Test that a valid filename is loaded as a moving texture correctly."""
    assert [
        texture.name.split("\\")[-1] for texture in load_moving_texture("floor.png")
    ] == [
        "floor.png-0-0-0-0-False-False-False-Simple ",
        "floor.png-0-0-0-0-True-False-False-Simple ",
    ]


def test_load_moving_texture_invalid_filename() -> None:
    """Test that an invalid filename is not loaded as a moving texture."""
    with pytest.raises(expected_exception=FileNotFoundError):
        load_moving_texture("temp.png")


def test_load_non_moving_texture_valid_filename() -> None:
    """Test that a valid filename is loaded as a non-moving texture correctly."""
    assert (
        load_non_moving_texture("floor.png").name.split("\\")[-1]
        == "floor.png-0-0-0-0-False-False-False-Simple "
    )


def test_load_non_moving_texture_invalid_filename() -> None:
    """Test that an invalid filename is not loaded asa a non-moving texture."""
    with pytest.raises(expected_exception=FileNotFoundError):
        load_non_moving_texture("temp.png")
