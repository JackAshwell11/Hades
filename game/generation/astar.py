"""Calculates the shortest path from one point to another using the A* algorithm."""
from __future__ import annotations

# Builtin
from heapq import heappop, heappush
from typing import TYPE_CHECKING

# Pip
import numpy as np

# Custom
from game.constants.generation import TileType
from game.generation.primitives import Point

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ("calculate_astar_path",)


offsets: list[tuple[int, int]] = [
    (0, -1),
    (-1, 0),
    (1, 0),
    (0, 1),
]


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


def get_neighbours(
    target: Point, height: int, width: int
) -> Generator[Point, None, None]:
    """Get the north, south, east and west neighbours of a given point if possible.

    Parameters
    ----------
    target: Point
        The point to get neighbours for.
    height: int
        The height of the grid.
    width: int
        The width of the grid.

    Returns
    -------
    Generator[Point, None, None]
        The given point's neighbours.
    """
    # Loop over each offset and get the grid position
    for dx, dy in offsets:
        x, y = target.x + dx, target.y + dy

        # Check if the grid position is within the bounds of the grid
        if (0 <= x < width) and (0 <= y < height):
            # Neighbour is valid so yield it
            yield Point(x, y)


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
            while current in came_from:
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
        for neighbour in get_neighbours(current, *grid.shape):
            if neighbour not in came_from:
                # Store the neighbour's parent and calculate its distance from the start
                # point
                came_from[neighbour] = current
                distances[neighbour] = distances[came_from[neighbour]] + 1

                # Check if the neighbour is an obstacle
                if grid[neighbour.y][neighbour.x] is TileType.OBSTACLE:
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
