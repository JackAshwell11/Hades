"""Tests all functions in generation/primitives.py."""
from __future__ import annotations

# Pip
import numpy as np
import pytest

# Custom
from game.constants.generation import TileType
from game.generation.primitives import Point, Rect

__all__ = ()


def test_point() -> None:
    """Test the Point class in primitives.py."""
    temp_point_one = Point(0, 0)
    assert temp_point_one == (0, 0)
    temp_point_two = Point("test", "test")  # type: ignore
    assert temp_point_two == ("test", "test")


def test_rect_init(
    valid_point_one: Point,
    valid_point_two: Point,
    invalid_point: Point,
    grid: np.ndarray,
) -> None:
    """Test the initialisation of the Rect class in primitives.py.

    Parameters
    ----------
    valid_point_one: Point
        The first valid point used for testing.
    valid_point_two: Point
        The second valid point used for testing.
    invalid_point: Point
        An invalid point used for testing.
    grid: np.ndarray
        The 2D grid used for testing.
    """
    temp_rect_one = Rect(grid, valid_point_one, valid_point_two)
    assert (
        temp_rect_one
        == (
            grid,
            valid_point_one,
            valid_point_two,
        )
        and repr(temp_rect_one)
        == "<Rect (Top left position=Point(x=3, y=5)) (Bottom right position=Point(x=5,"
        " y=7)) (Center position=Point(x=4, y=6)) (Width=2) (Height=2)>"
        and temp_rect_one.width == 2
        and temp_rect_one.height == 2
        and temp_rect_one.center == Point(4, 6)
    )
    temp_rect_two = Rect(grid, valid_point_one, invalid_point)
    assert temp_rect_two == (
        grid,
        valid_point_one,
        invalid_point,
    )
    with pytest.raises(TypeError):
        repr(temp_rect_two)


def test_rect_properties(rect: Rect) -> None:
    """Test all the properties in the Rect class.

    Parameters
    ----------
    rect: Rect
        The rect used for testing.
    """
    assert (
        rect.width == 2
        and rect.height == 2
        and rect.center_x == 4
        and rect.center_y == 6
        and rect.center == (4, 6)
    )


def test_rect_get_distance_to(
    valid_point_one: Point,
    boundary_point: Point,
    invalid_point: Point,
    rect: Rect,
    grid: np.ndarray,
) -> None:
    """Test the get_distance_to function in the Rect class.

    Parameters
    ----------
    valid_point_one: Point
        The first valid point used for testing.
    boundary_point: Point
        A boundary point used for testing.
    invalid_point: Point
        An invalid point used for testing.
    rect: Rect
        The rect used for testing.
    grid: np.ndarray
        The 2D grid used for testing.
    """
    temp_rect_one = Rect(grid, valid_point_one, boundary_point)
    assert temp_rect_one.get_distance_to(rect) == 4.47213595499958
    temp_rect_two = Rect(grid, valid_point_one, invalid_point)
    with pytest.raises(TypeError):
        temp_rect_two.get_distance_to(rect)


def test_rect_place_rect(rect: Rect) -> None:
    """Test the place_rect function in the Rect class.

    Parameters
    ----------
    rect: Rect
        The rect used for testing.
    """
    # Check if the place_rect function places a rect inside the 2D array
    rect.place_rect()
    assert (
        np.all(
            rect.grid[
                rect.top_left.y : rect.bottom_right.y + 1,
                rect.top_left.x : rect.bottom_right.x + 1,
            ]
            != TileType.EMPTY
        )
        and rect.grid[rect.center_y][rect.center_x] == TileType.FLOOR
    )
