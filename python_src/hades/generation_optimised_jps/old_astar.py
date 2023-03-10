"""Calculates the shortest path from one point to another using the A* algorithm."""
from __future__ import annotations

# Builtin
from heapq import heappop, heappush
from typing import TYPE_CHECKING

# Pip
import numpy as np

# Custom
from hades.constants.generation import TileType
from hades.generation.primitives import Point

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ("calculate_astar_path",)


def grid_bfs(
    target: tuple[int, int],
    height: int,
    width: int,
) -> Generator[tuple[int, int], None, None]:
    """Get a target's neighbours based on a given list of offsets.
    Note that this uses the same logic as the grid_bfs() function in the C++ extensions,
    however, it is much slower due to Python.
    Parameters
    ----------
    target: tuple[int, int]
        The target to get neighbours for.
    height: int
        The height of the grid.
    width: int
        The width of the grid.
    Returns
    -------
    Generator[tuple[int, int], None, None]
        A list of the target's neighbours.
    """
    # Get all the neighbour floor tile positions relative to the current target
    for dx, dy in (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ):
        # Check if the neighbour position is within the boundaries of the grid or not
        x, y = target[0] + dx, target[1] + dy
        if (x < 0 or x >= width) or (y < 0 or y >= height):
            continue

        # Yield the neighbour tile position
        yield x, y


def heuristic(a: Point, b: Point) -> int:
    """Calculate the Manhattan distance between two points.
     This preferable to the Euclidean distance since we can generate staircase-like
     paths instead of straight line paths.
     Further reading which may be useful:
     `Manhattan distance <https://en.wikipedia.org/wiki/Taxicab_geometry>`_
     `Euclidean distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_
    Parameters
    ----------
    a: Point
        The first point.
    b: Point
        The second point.
    """
    return abs(a.x - b.x) + abs(a.y - b.y)


def calculate_astar_path(grid: np.ndarray, start: Point, end: Point) -> list[Point]:
    """Calculate the shortest path from one point to another using the A* algorithm.
    Further reading which may be useful:
    `The A* algorithm <https://en.wikipedia.org/wiki/A*_search_algorithm>`_
    Parameters
    ----------
    grid: np.ndarray
        The 2D grid which represents the dungeon.
    start: Point
        The start point for the algorithm.
    end: Point
        The end point for the algorithm.
    Returns
    -------
    list[Point]
        A list of points mapping out the shortest path from start to end.
    """
    # Initialise a few variables needed for the pathfinding
    heap: list[tuple[int, Point]] = [(0, start)]
    came_from: dict[Point, Point] = {start: start}
    distances: dict[Point, int] = {start: 0}
    total_costs: dict[Point, int] = {start: 0}

    # Loop until the heap is empty
    while heap:
        # Get the lowest-cost point from the heap
        _, current = heappop(heap)

        # Check if we've reached our target
        if current == end:
            # Backtrack through came_from to get the path and return the result
            result = []
            while True:
                # Add the path to the result list
                result.append(current)

                # Test if we've hit the starting point
                if came_from[current] != current:
                    current = came_from[current]
                else:
                    break
            return result

        # Add all the neighbours to the heap with their cost being f = g + h:
        #   f - The total cost of traversing the neighbour.
        #   g - The distance between the start point and the neighbour point.
        #   h - The estimated distance from the neighbour point to the end point.
        for bfs_neighbour in grid_bfs(current, *grid.shape):
            if bfs_neighbour not in came_from:
                # Store the neighbour's parent and calculate its distance from the start
                # point
                neighbour = Point(*bfs_neighbour)
                came_from[neighbour] = current
                distances[neighbour] = distances[came_from[neighbour]] + 1

                # Check if the neighbour is an obstacle
                if grid[neighbour.y][neighbour.x] == TileType.OBSTACLE:
                    # Set the total cost for the obstacle to infinity
                    total_costs[neighbour] = np.inf
                else:
                    # Set the total cost for the neighbour to f = g + h
                    total_costs[neighbour] = distances[neighbour] + heuristic(
                        current, neighbour
                    )

                # Add the neighbour to the heap
                heappush(heap, (total_costs[neighbour], neighbour))

    # A path can't be found
    return []
