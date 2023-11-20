"""Tests all classes and functions in textures.py."""

from __future__ import annotations

# Pip
import pytest

# Custom
from hades.textures import (
    BiggerThanError,
    grid_pos_to_pixel,
    load_moving_texture,
    load_non_moving_texture,
)

__all__ = ()


def test_raise_bigger_than_error() -> None:
    """Test that BiggerThanError is raised correctly."""
    with pytest.raises(
        expected_exception=BiggerThanError,
        match="The input must be bigger than or equal to 10.",
    ):
        raise BiggerThanError(min_value=10)


def test_grid_pos_to_pixel_positive() -> None:
    """Test that a valid position is converted correctly."""
    assert grid_pos_to_pixel(500, 500) == (32032.0, 32032.0)


def test_grid_pos_to_pixel_zero() -> None:
    """Test that a zero position is converted correctly."""
    assert grid_pos_to_pixel(0, 0) == (32.0, 32.0)


def test_grid_pos_to_pixel_negative() -> None:
    """Test that a negative position is converted correctly."""
    with pytest.raises(
        expected_exception=BiggerThanError,
        match="The input must be bigger than or equal to 0.",
    ):
        grid_pos_to_pixel(-500, -500)


def test_grid_pos_to_pixel_string() -> None:
    """Test that a position made of strings is converted correctly."""
    with pytest.raises(expected_exception=TypeError):
        grid_pos_to_pixel("test", "test")  # type: ignore[arg-type]


def test_load_moving_texture_valid_filename() -> None:
    """Test that a valid filename is loaded as a moving texture correctly."""
    assert [texture.cache_name for texture in load_moving_texture("floor.png")] == [
        (
            "b3d8c789f0ab79a64f6ee6c8eac8fc329b53a3a56ed6c0ee262522cefef5dcf4|(0, 1, 2,"
            " 3)|SimpleHitBoxAlgorithm|"
        ),
        (
            "b3d8c789f0ab79a64f6ee6c8eac8fc329b53a3a56ed6c0ee262522cefef5dcf4|(1, 0, 3,"
            " 2)|SimpleHitBoxAlgorithm|"
        ),
    ]


def test_load_moving_texture_invalid_filename() -> None:
    """Test that an invalid filename is not loaded as a moving texture."""
    with pytest.raises(expected_exception=FileNotFoundError):
        load_moving_texture("temp.png")


def test_load_non_moving_texture_valid_filename() -> None:
    """Test that a valid filename is loaded as a non-moving texture correctly."""
    assert (
        load_non_moving_texture("floor.png").cache_name
        == "b3d8c789f0ab79a64f6ee6c8eac8fc329b53a3a56ed6c0ee262522cefef5dcf4|(0, 1, 2,"
        " 3)|SimpleHitBoxAlgorithm|"
    )


def test_load_non_moving_texture_invalid_filename() -> None:
    """Test that an invalid filename is not loaded asa a non-moving texture."""
    with pytest.raises(expected_exception=FileNotFoundError):
        load_non_moving_texture("temp.png")
