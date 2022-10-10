"""Calculates the shortest path from one point to another using the A* algorithm."""
from __future__ import annotations

# Builtin
from heapq import heappop, heappush
from typing import TYPE_CHECKING

# Custom
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


def is_obstacle(grid: np.ndarray, x: int, y: int) -> bool:
    try:
        return grid[y][x] == -1
    except IndexError:
        return False


def find_neighbours(
    grid: np.ndarray, current: Point, parent: Point | None
) -> Generator[tuple[int, int], None, None]:
    if parent:
        dx, dy = (
            int((current.x - parent.x) / max(abs(current.x - parent.x), 1)),
            int((current.y - parent.y) / max(abs(current.y - parent.y), 1)),
        )

        if dx != 0 and dy != 0:
            x_mods = (dx, 0, dx)
            y_mods = (0, dy, dy)
        elif dx != 0:
            x_mods = (0, 0, dx)
            y_mods = (-1, 1, 0)
        else:
            x_mods = (-1, 1, 0)
            y_mods = (0, 0, dy)
        for x_mod, y_mod in zip(x_mods, y_mods):
            new_pnt = current.x + x_mod, current.y + y_mod
            if not is_obstacle(grid, *new_pnt):
                yield new_pnt
    else:
        yield from grid_bfs(current, *grid.shape)


def jump(
    grid: np.ndarray, current_point: Point, end: Point, parent_point: Point | None
) -> Point | None:
    if current_point == end:
        return current_point

    if (
        0 > current_point.x
        or current_point.x >= grid.shape[1]
        or 0 > current_point.y
        or current_point.y >= grid.shape[0]
        or is_obstacle(grid, *current_point)
    ):
        print("out of bounds or obstacle")
        return None

    print("jump")
    print(f"current={current_point} ({grid[current_point.y][current_point.x]})")
    dx, dy = current_point.x - parent_point.x, current_point.y - parent_point.y

    if dx != 0 and dy != 0:
        print("diagonal")
        if jump(
            grid, Point(current_point.x + dx, current_point.y), end, current_point
        ) or jump(
            grid, Point(current_point.x, current_point.y + dy), end, current_point
        ):
            return current_point
    elif dx != 0:
        print(f"horizontal dx={dx}, dy={dy}")
        if (
            not is_obstacle(grid, current_point.x, current_point.y + 1)
            and is_obstacle(grid, current_point.x - dx, current_point.y + 1)
        ) or (
            not is_obstacle(grid, current_point.x, current_point.y - 1)
            and is_obstacle(grid, current_point.x - dx, current_point.y - 1)
        ):
            return current_point
    else:
        print(f"vertical dx={dx}, dy={dy}")
        if (
            not is_obstacle(grid, current_point.x + 1, current_point.y)
            and is_obstacle(grid, current_point.x + 1, current_point.y - dy)
        ) or (
            not is_obstacle(grid, current_point.x - 1, current_point.y)
            and is_obstacle(grid, current_point.x - 1, current_point.y - dy)
        ):
            return current_point

    return jump(
        grid, Point(current_point.x + dx, current_point.y + dy), end, current_point
    )


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
        print(f"current={current}")

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
            print(f"neighbour={neighbour}")
            print(f"neighbour val={grid[neighbour[1]][neighbour[0]]}")
            jump_point = jump(grid, Point(*neighbour), end, current)
            if jump_point is not None and jump_point not in came_from:
                print(f"jump point={jump_point} ({grid[jump_point.y][jump_point.x]})")
                # Store the jump_point's parent
                came_from[jump_point] = current

                # Calculate the total cost for traversing this jump point
                distances[jump_point] = distances[came_from[jump_point]] + 1
                f_cost = distances[jump_point] + heuristic(current, jump_point)

                # Add this jump point to the heap
                heappush(heap, (f_cost, jump_point, current))

    # A path can't be found
    return []


# TODO: SEEMS LIKE THE CODE MISSES THE FIRST JUMP POINT FOR SOME REASON
