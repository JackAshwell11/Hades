"""Holds fixtures and test data used by all tests."""
from __future__ import annotations

# Pip
import arcade
import numpy as np
import pytest

# Custom
from hades.constants.generation import TileType
from hades.extensions import VectorField
from hades.generation.bsp import Leaf
from hades.generation.map import Map
from hades.generation.primitives import Point, Rect

__all__ = ()


@pytest.fixture
def valid_point_one() -> Point:
    """Initialise the first valid point for use in testing.

    Returns
    -------
    Point
        The first valid point used for testing.
    """
    return Point(3, 5)


@pytest.fixture
def valid_point_two() -> Point:
    """Initialise the second valid point for use in testing.

    Returns
    -------
    Point
        The second valid point used for testing.
    """
    return Point(5, 7)


@pytest.fixture
def boundary_point() -> Point:
    """Initialise a boundary point for use in testing.

    Returns
    -------
    Point
        The boundary point used for testing.
    """
    return Point(0, 0)


@pytest.fixture
def invalid_point() -> Point:
    """Initialise an invalid point for use in testing.

    Returns
    -------
    Point
        The invalid point used for testing.
    """
    return Point("test", "test")  # type: ignore


@pytest.fixture
def grid() -> np.ndarray:
    """Initialise a 2D numpy grid for use in testing.

    Returns
    -------
    np.ndarray
        The 2D numpy grid used for testing.
    """
    return np.full((50, 50), TileType.EMPTY, np.int8)


@pytest.fixture
def rect(valid_point_one: Point, valid_point_two: Point, grid: np.ndarray) -> Rect:
    """Initialise a rect for use in testing.

    Parameters
    ----------
    valid_point_one: Point
        The first valid point used for testing.
    valid_point_two: Point
        The second valid point used for testing.
    grid: np.ndarray
        The 2D grid used for testing.

    Returns
    -------
    Rect
        The rect used for testing.
    """
    return Rect(grid, valid_point_one, valid_point_two)


@pytest.fixture
def leaf(boundary_point: Point, grid: np.ndarray) -> Leaf:
    """Initialise a leaf for use in testing.

    Parameters
    ----------
    boundary_point: Point
        A boundary point used for testing.
    grid: np.ndarray
        The 2D grid used for testing.

    Returns
    -------
        The leaf used for testing.
    """
    return Leaf(boundary_point, Point(grid.shape[1], grid.shape[0]), grid)


@pytest.fixture
def map_obj() -> Map:
    """Initialise a map for use in testing.

    Returns
    -------
    Map
        The map used for testing.
    """
    return Map(0)


@pytest.fixture
def walls() -> arcade.SpriteList:
    """Initialise the walls spritelist for use in testing.

    Returns
    -------
    arcade.SpriteList
        The walls spritelist for use in testing.
    """
    temp_spritelist = arcade.SpriteList()
    temp_spritelist.extend(
        [
            arcade.Sprite(center_x=84.0, center_y=84.0),
            arcade.Sprite(center_x=140.0, center_y=84.0),
            arcade.Sprite(center_x=196.0, center_y=84.0),
            arcade.Sprite(center_x=252.0, center_y=84.0),
            arcade.Sprite(center_x=308.0, center_y=84.0),
            arcade.Sprite(center_x=364.0, center_y=84.0),
            arcade.Sprite(center_x=420.0, center_y=84.0),
            arcade.Sprite(center_x=84.0, center_y=140.0),
            arcade.Sprite(center_x=420.0, center_y=140.0),
            arcade.Sprite(center_x=812.0, center_y=140.0),
            arcade.Sprite(center_x=868.0, center_y=140.0),
            arcade.Sprite(center_x=924.0, center_y=140.0),
            arcade.Sprite(center_x=980.0, center_y=140.0),
            arcade.Sprite(center_x=1036.0, center_y=140.0),
            arcade.Sprite(center_x=84.0, center_y=196.0),
            arcade.Sprite(center_x=420.0, center_y=196.0),
            arcade.Sprite(center_x=812.0, center_y=196.0),
            arcade.Sprite(center_x=1036.0, center_y=196.0),
            arcade.Sprite(center_x=84.0, center_y=252.0),
            arcade.Sprite(center_x=420.0, center_y=252.0),
            arcade.Sprite(center_x=812.0, center_y=252.0),
            arcade.Sprite(center_x=1036.0, center_y=252.0),
            arcade.Sprite(center_x=84.0, center_y=308.0),
            arcade.Sprite(center_x=420.0, center_y=308.0),
            arcade.Sprite(center_x=812.0, center_y=308.0),
            arcade.Sprite(center_x=1036.0, center_y=308.0),
            arcade.Sprite(center_x=84.0, center_y=364.0),
            arcade.Sprite(center_x=420.0, center_y=364.0),
            arcade.Sprite(center_x=812.0, center_y=364.0),
            arcade.Sprite(center_x=1036.0, center_y=364.0),
            arcade.Sprite(center_x=84.0, center_y=420.0),
            arcade.Sprite(center_x=420.0, center_y=420.0),
            arcade.Sprite(center_x=756.0, center_y=420.0),
            arcade.Sprite(center_x=812.0, center_y=420.0),
            arcade.Sprite(center_x=1036.0, center_y=420.0),
            arcade.Sprite(center_x=84.0, center_y=476.0),
            arcade.Sprite(center_x=420.0, center_y=476.0),
            arcade.Sprite(center_x=756.0, center_y=476.0),
            arcade.Sprite(center_x=980.0, center_y=476.0),
            arcade.Sprite(center_x=1036.0, center_y=476.0),
            arcade.Sprite(center_x=84.0, center_y=532.0),
            arcade.Sprite(center_x=420.0, center_y=532.0),
            arcade.Sprite(center_x=700.0, center_y=532.0),
            arcade.Sprite(center_x=756.0, center_y=532.0),
            arcade.Sprite(center_x=980.0, center_y=532.0),
            arcade.Sprite(center_x=84.0, center_y=588.0),
            arcade.Sprite(center_x=140.0, center_y=588.0),
            arcade.Sprite(center_x=364.0, center_y=588.0),
            arcade.Sprite(center_x=420.0, center_y=588.0),
            arcade.Sprite(center_x=700.0, center_y=588.0),
            arcade.Sprite(center_x=980.0, center_y=588.0),
            arcade.Sprite(center_x=140.0, center_y=644.0),
            arcade.Sprite(center_x=364.0, center_y=644.0),
            arcade.Sprite(center_x=588.0, center_y=644.0),
            arcade.Sprite(center_x=644.0, center_y=644.0),
            arcade.Sprite(center_x=700.0, center_y=644.0),
            arcade.Sprite(center_x=924.0, center_y=644.0),
            arcade.Sprite(center_x=980.0, center_y=644.0),
            arcade.Sprite(center_x=1036.0, center_y=644.0),
            arcade.Sprite(center_x=140.0, center_y=700.0),
            arcade.Sprite(center_x=364.0, center_y=700.0),
            arcade.Sprite(center_x=588.0, center_y=700.0),
            arcade.Sprite(center_x=1036.0, center_y=700.0),
            arcade.Sprite(center_x=28.0, center_y=756.0),
            arcade.Sprite(center_x=84.0, center_y=756.0),
            arcade.Sprite(center_x=140.0, center_y=756.0),
            arcade.Sprite(center_x=364.0, center_y=756.0),
            arcade.Sprite(center_x=420.0, center_y=756.0),
            arcade.Sprite(center_x=476.0, center_y=756.0),
            arcade.Sprite(center_x=532.0, center_y=756.0),
            arcade.Sprite(center_x=588.0, center_y=756.0),
            arcade.Sprite(center_x=1036.0, center_y=756.0),
            arcade.Sprite(center_x=28.0, center_y=812.0),
            arcade.Sprite(center_x=1036.0, center_y=812.0),
            arcade.Sprite(center_x=28.0, center_y=868.0),
            arcade.Sprite(center_x=1036.0, center_y=868.0),
            arcade.Sprite(center_x=28.0, center_y=924.0),
            arcade.Sprite(center_x=1036.0, center_y=924.0),
            arcade.Sprite(center_x=28.0, center_y=980.0),
            arcade.Sprite(center_x=420.0, center_y=980.0),
            arcade.Sprite(center_x=476.0, center_y=980.0),
            arcade.Sprite(center_x=532.0, center_y=980.0),
            arcade.Sprite(center_x=588.0, center_y=980.0),
            arcade.Sprite(center_x=1036.0, center_y=980.0),
            arcade.Sprite(center_x=28.0, center_y=1036.0),
            arcade.Sprite(center_x=84.0, center_y=1036.0),
            arcade.Sprite(center_x=140.0, center_y=1036.0),
            arcade.Sprite(center_x=196.0, center_y=1036.0),
            arcade.Sprite(center_x=252.0, center_y=1036.0),
            arcade.Sprite(center_x=308.0, center_y=1036.0),
            arcade.Sprite(center_x=364.0, center_y=1036.0),
            arcade.Sprite(center_x=420.0, center_y=1036.0),
            arcade.Sprite(center_x=588.0, center_y=1036.0),
            arcade.Sprite(center_x=644.0, center_y=1036.0),
            arcade.Sprite(center_x=700.0, center_y=1036.0),
            arcade.Sprite(center_x=756.0, center_y=1036.0),
            arcade.Sprite(center_x=812.0, center_y=1036.0),
            arcade.Sprite(center_x=868.0, center_y=1036.0),
            arcade.Sprite(center_x=924.0, center_y=1036.0),
            arcade.Sprite(center_x=980.0, center_y=1036.0),
            arcade.Sprite(center_x=1036.0, center_y=1036.0),
        ]
    )
    return temp_spritelist


@pytest.fixture
def vector_field(walls: arcade.SpriteList) -> VectorField:
    """Initialise a vector field for use in testing.

    Parameters
    ----------
    walls: arcade.SpriteList
        The walls spritelist used for testing.

    Returns
    -------
    VectorField
        The vector field for use in testing.
    """
    return VectorField(walls, 20, 20)
