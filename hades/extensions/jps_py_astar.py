"""Calculates the shortest path from one point to another using the A* algorithm."""
from __future__ import annotations

from heapq import heappop, heappush

# Builtin
from itertools import count
from collections.abc import Generator

# Pip
import numpy as np

# Custom
from hades.common import grid_bfs
from hades.constants.generation import TileType
from hades.generation.primitives import Point

__all__ = ("calculate_astar_path",)


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


def is_within_bounds(x: int, y: int, height: int, width: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def is_obstacle(grid: np.ndarray, x: int, y: int) -> bool:
    return is_within_bounds(x, y, *grid.shape) and grid[y][x] == TileType.OBSTACLE


def is_traversable(grid: np.ndarray, x: int, y: int) -> bool:
    return is_within_bounds(x, y, *grid.shape) and not is_obstacle(grid, x, y)


def find_neighbours(
    grid: np.ndarray, current: Point, parent: Point | None
) -> Generator[tuple[int, int], None, None]:
    if parent is None:
        for i in list(grid_bfs(current, *grid.shape)):
            yield i
    else:
        dx, dy = (current.x - parent.x) // max(abs(current.x - parent.x), 1), (current.y - parent.y) // max(abs(current.y - parent.y), 1)
        if dx != 0:
            first_x, first_y = current.x, current.y - 1
            second_x, second_y = current.x, current.y + 1
            third_x, third_y = current.x + dx, current.y
        elif dy != 0:
            first_x, first_y = current.x - 1, current.y
            second_x, second_y = current.x + 1, current.y
            third_x, third_y = current.x, current.y + dy

        if not is_obstacle(grid, first_x, first_y):
            yield first_x, first_y
        if not is_obstacle(grid, second_x, second_y):
            yield second_x, second_y
        if not is_obstacle(grid, third_x, third_y):
            yield third_x, third_y


def jump(grid: np.ndarray, current: Point, end: Point, parent: Point | None) -> Point | None:
    if current == end:
        return current

    if not is_traversable(grid, current.x, current.y):
        return None

    dx, dy = current.x - parent.x, current.y - parent.y
    if dx != 0:
        if (is_traversable(grid, current.x, current.y + 1) and not is_traversable(grid, current.x - dx, current.y + 1)) or (is_traversable(grid, current.x, current.y - 1) and not is_traversable(grid, current.x - dx, current.y - 1)):
            return current
    elif dy != 0:
        if (is_traversable(grid, current.x + 1, current.y) and not is_traversable(grid, current.x + 1, current.y - dy)) or (is_traversable(grid, current.x - 1, current.y) and not is_traversable(grid, current.x - 1, current.y - dy)):
            return current

        if (jump(grid, Point(current.x + 1, current.y), end, current) is not None) or (jump(grid, Point(current.x - 1, current.y), end, current) is not None):
            return current
    else:
        return None

    return jump(grid, Point(current.x + dx, current.y + dy), end, current)


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
    heap: list[tuple[int, Point, Point | None]] = [(0, start, None)]
    came_from: dict[Point, Point] = {start: start}
    distances: dict[Point, int] = {start: 0}

    # Loop until the heap is empty
    while heap:
        # Get the lowest-cost point from the heap
        _, current, parent = heappop(heap)

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

        for neighbour in find_neighbours(grid, current, parent):
            # Add the jump point to the heap with their cost being f = g + h:
            #   f - The total cost of traversing the jump point.
            #   g - The distance between the start point and the jump point.
            #   h - The estimated distance from the jump point to the end point.
            jump_point = jump(grid, Point(*neighbour), end, current)
            if jump_point is not None and jump_point not in came_from:
                # Store the jump_point's parent
                came_from[jump_point] = current

                # Calculate the total cost for traversing this jump point
                distances[jump_point] = distances[came_from[jump_point]] + 1
                f_cost = distances[jump_point] + heuristic(current, jump_point)

                # Add this jump point to the heap
                heappush(heap, (f_cost, jump_point, current))

    # A path can't be found
    return []
