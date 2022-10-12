"""Calculates the shortest path from one point to another using the A* algorithm."""
from __future__ import annotations

# Builtin
from collections import deque
from heapq import heappop, heappush
from typing import TYPE_CHECKING

# Custom
from hades.constants.generation import TileType
from hades.generation.primitives import Point

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np


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


def find_neighbours(
    grid: np.ndarray, current: Point, parent: Point | None
) -> Generator[tuple[int, int], None, None]:
    """Find all the neighbours based on the current point and the direction."""
    # Test if origin is the starting point
    if parent:
        # Origin is not the starting point so calculate the direction
        dx, dy = (
            int((current.x - parent.x) / max(abs(current.x - parent.x), 1)),
            int((current.y - parent.y) / max(abs(current.y - parent.y), 1)),
        )

        # Return all neighbours in the front of the origin based on the direction we're
        # moving in
        if dx != 0 and dy != 0:
            x_mods = (dx, 0, dx)
            y_mods = (0, dy, dy)
        elif dx != 0:
            x_mods = (dx, dx, dx)
            y_mods = (0, 1, -1)
        else:
            x_mods = (0, 1, -1)
            y_mods = (dy, dy, dy)
        yield from (
            (current.x + x_mod, current.y + y_mod)
            for x_mod, y_mod in zip(x_mods, y_mods)
        )
    else:
        # Origin is starting point so return all neighbours
        yield from grid_bfs(current, *grid.shape)


def walkable(grid: np.ndarray, x: int, y: int) -> bool:
    """Determine if a point exists in the grid and is not an obstacle."""
    return (
        0 <= x < grid.shape[1]
        and 0 <= y < grid.shape[0]
        and grid[y][x] != TileType.OBSTACLE
    )


def jump(grid: np.ndarray, origin: Point, parent: Point, end: Point) -> Point | None:
    """Determine the next jump point based on the current point."""
    stack: deque[tuple[Point, Point]] = deque((origin, parent))

    while stack:
        current, current_parent = stack.pop()
        if not walkable(grid, *current):
            continue

        if current == end:
            return origin

        dx, dy = current.x - current_parent.x, current.y - current_parent.y
        if dx != 0 and dy != 0:
            stack.append((Point(current.x + dx, current.y), current))
            stack.append((Point(current.x, current.y + dy), current))
        elif dx != 0:
            if (
                not walkable(grid, current.x - dx, current.y - 1)
                and walkable(grid, current.x, current.y - 1)
            ) or (
                not walkable(grid, current.x - dx, current.y + 1)
                and walkable(grid, current.x, current.y + 1)
            ):
                return origin
        else:
            if (
                not walkable(grid, current.x - 1, current.y - dy)
                and walkable(grid, current.x - 1, current.y)
            ) or (
                not walkable(grid, current.x + 1, current.y - dy)
                and walkable(grid, current.x + 1, current.y)
            ):
                return origin

        if walkable(grid, current.x + dx, current.y) and walkable(
            grid, current.x, current.y + dy
        ):
            stack.append((Point(current.x + dx, current.y + dy), current))
        else:
            return None

    return None


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
            jump_point = jump(grid, Point(*neighbour), current, end)
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
