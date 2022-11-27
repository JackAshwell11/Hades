"""Tests all functions in extensions/astar."""
from __future__ import annotations

import math

# Builtin
import random
from typing import TYPE_CHECKING

# Pip
import numpy as np
import pytest

# Custom
from hades.constants.generation import TileType
from hades.extensions import calculate_astar_path

if TYPE_CHECKING:
    from hades.generation.primitives import Point

__all__ = ()


def get_obstacle_grid() -> np.ndarray:
    """Get a 2D numpy grid with some obstacles for use in testing.

    Returns
    -------
    np.ndarray
        The 2D numpy grid.
    """
    # Create a temporary grid with some obstacles. This uses the same code from
    # Map.create_hallways()
    temp_grid = np.full(
        (10, 10),
        TileType.EMPTY,  # type: ignore
        TileType,
    )
    obstacle_positions = list(zip(*np.where(temp_grid == TileType.EMPTY)))
    for _ in range(25):
        temp_grid[
            obstacle_positions.pop(random.randrange(len(obstacle_positions)))
        ] = TileType.OBSTACLE
    return temp_grid


def heuristic(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two pairs.

    This uses a different heuristic to the C++ extension since in some cases the
    distance can be Euclidean instead of Manhattan (diagonal line of sight between start
    and end) so we need to get the minimum size that a path can have.

    Parameters
    ----------
    a: Point
        The first point.
    b: Point
        The second point.

    Returns
    -------
    float
        The Euclidean distance.
    """
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def test_calculate_astar_path(
    valid_point_one: Point,
    valid_point_two: Point,
    boundary_point: Point,
    invalid_point: Point,
) -> None:
    """Test the calculate_astar_path function in astar.cpp.

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
    temp_result_one = calculate_astar_path(
        get_obstacle_grid(),
        valid_point_one,
        valid_point_two,
    )
    assert (
        temp_result_one[0] == valid_point_two
        and temp_result_one[-1] == valid_point_one
        and len(temp_result_one) >= heuristic(valid_point_one, valid_point_two)
    )
    temp_result_two = calculate_astar_path(
        get_obstacle_grid(),
        valid_point_one,
        boundary_point,
    )
    assert (
        temp_result_two[0] == boundary_point
        and temp_result_two[-1] == valid_point_one
        and len(temp_result_two) >= heuristic(valid_point_one, boundary_point)
    )
    with pytest.raises(SystemError):
        calculate_astar_path(
            get_obstacle_grid(),
            valid_point_one,
            invalid_point,
        )
