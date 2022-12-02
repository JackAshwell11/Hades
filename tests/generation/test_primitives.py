"""Tests all functions in generation/primitives.py."""
from __future__ import annotations

# Pip
import numpy as np
import numpy.typing as npt
import pytest

# Custom
from hades.constants.generation import TileType
from hades.generation.primitives import Point, Rect

__all__ = ()


def test_point() -> None:
    """Test the Point class in primitives.py."""
    assert Point(0, 0) == (0, 0)
    assert Point("test", "test") == ("test", "test")  # type: ignore


def test_rect_init(
    valid_point_one: Point,
    valid_point_two: Point,
    invalid_point: Point,
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
    """
    temp_rect_one = Rect(valid_point_one, valid_point_two)
    assert (
        temp_rect_one
        == (
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
    temp_rect_two = Rect(valid_point_one, invalid_point)
    assert temp_rect_two == (
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
    """
    assert (
        Rect(valid_point_one, boundary_point).get_distance_to(rect)
        == 2.8284271247461903
    )
    with pytest.raises(TypeError):
        Rect(valid_point_one, invalid_point).get_distance_to(rect)


def test_rect_place_rect(rect: Rect, grid: npt.NDArray[np.int8]) -> None:
    """Test the place_rect function in the Rect class.

    Parameters
    ----------
    rect: Rect
        The rect used for testing.
    grid: npt.NDArray[np.int8]
        The 2D grid used for testing.
    """
    # Check if the place_rect function places a rect inside the 2D array
    rect.place_rect(grid)
    assert (
        np.all(
            grid[
                rect.top_left.y : rect.bottom_right.y + 1,
                rect.top_left.x : rect.bottom_right.x + 1,
            ]
            != TileType.EMPTY
        )
        and grid[rect.center_y][rect.center_x] == TileType.FLOOR
    )
