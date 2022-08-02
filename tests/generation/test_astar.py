"""Tests all functions in generation/astar.py."""
from __future__ import annotations

# Pip
import numpy as np
import pytest

# Custom
from game.constants.generation import TileType
from game.generation import astar
from game.generation.primitives import Point

__all__ = ()


@pytest.fixture
def valid_point_one() -> Point:
    """Initialise the first valid point for use in testing.

    Returns
    -------
    Point
        The first valid point.
    """
    return Point(1, 3)


@pytest.fixture
def valid_point_two() -> Point:
    """Initialise the second valid point for use in testing.

    Returns
    -------
    Point
        The second valid point.
    """
    return Point(5, 6)


@pytest.fixture
def boundary_point() -> Point:
    """Initialise a boundary point for use in testing.

    Returns
    -------
    Point
        The boundary point.
    """
    return Point(0, 0)


@pytest.fixture
def invalid_point() -> Point:
    """Initialise an invalid point for use in testing.

    Returns
    -------
    Point
        The invalid point.
    """
    return Point("test", "test")  # type: ignore


def get_grid() -> np.ndarray:
    """Get a 2D numpy grid with some sample data for use in testing.

    Returns
    -------
    np.ndarray
        The 2D numpy grid.
    """
    # Create a temporary grid with some obstacles. This uses the same code from
    # Map._create_hallways
    temp_grid = np.full(
        (8, 10),
        TileType.EMPTY,
        np.int8,
    )
    y, x = np.where(temp_grid == TileType.EMPTY)
    arr_index = np.random.choice(len(y), 10)
    temp_grid[y[arr_index], x[arr_index]] = TileType.OBSTACLE
    return temp_grid


def test_heuristic(
    valid_point_one: Point, boundary_point: Point, invalid_point: Point
) -> None:
    """Tests the heuristic function in astar.py.

    Parameters
    ----------
    valid_point_one: Point
        A valid point used for testing.
    boundary_point: Point
        A boundary point used for testing.
    invalid_point: Point
        An invalid point used for testing.
    """
    assert astar.heuristic(valid_point_one, boundary_point) == 4
    with pytest.raises(TypeError):
        astar.heuristic(invalid_point, invalid_point)


def test_get_neighbours(
    valid_point_one: Point, boundary_point: Point, invalid_point: Point
) -> None:
    """Tests the get_neighbours function in astar.py.

    Parameters
    ----------
    valid_point_one: Point
        A valid point used for testing.
    boundary_point: Point
        A boundary point used for testing.
    invalid_point: Point
        An invalid point used for testing.
    """
    assert len(list(astar.get_neighbours(valid_point_one, 10, 10))) == 4  # noqa
    assert len(list(astar.get_neighbours(boundary_point, 10, 10))) == 2  # noqa
    with pytest.raises(TypeError):
        list(astar.get_neighbours(invalid_point, 10, 10))  # noqa


def test_calculate_astar_path(
    valid_point_one: Point,
    valid_point_two: Point,
    boundary_point: Point,
    invalid_point: Point,
) -> None:
    """Tests the calculate_astar_path function in astar.py.

    Parameters
    ----------
    valid_point_one: Point
        The first valid point used for testing.
    valid_point_two: Point
        The second valid point used for testing.
    boundary_point: Point
        A boundary point used for testing.
    invalid_point: Point
        An invalid point used for testing.
    """
    temp_result_one = astar.calculate_astar_path(
        get_grid(), valid_point_one, valid_point_two
    )
    assert (
        temp_result_one[0] == valid_point_two
        and temp_result_one[-1] == valid_point_one
        and len(temp_result_one) >= astar.heuristic(valid_point_one, valid_point_two)
    )
    temp_result_two = astar.calculate_astar_path(
        get_grid(), valid_point_one, boundary_point
    )
    assert (
        temp_result_two[0] == boundary_point
        and temp_result_two[-1] == valid_point_one
        and len(temp_result_two) >= astar.heuristic(valid_point_one, boundary_point)
    )
    assert not astar.calculate_astar_path(get_grid(), valid_point_one, invalid_point)
