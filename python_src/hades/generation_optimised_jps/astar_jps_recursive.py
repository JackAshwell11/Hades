"""Calculates the shortest path from one point to another using the A* algorithm."""
from __future__ import annotations

# Builtin
from heapq import heappop, heappush
from typing import TYPE_CHECKING

# Custom
from hades.constants.generation import TileType
from hades.generation_optimised_jps.primitives import Point

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np

__all__ = ("calculate_astar_path",)

INTERCARDINAL_OFFSETS = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def walkable(grid: np.ndarray, x: int, y: int) -> bool:
    return (
        0 <= x < grid.shape[1]
        and 0 <= y < grid.shape[0]
        and grid[y][x] != TileType.OBSTACLE
    )


def jump(grid: np.ndarray, current: Point, parent: Point, end: Point) -> Point | None:
    if not walkable(grid, *current):
        return None

    if current == end:
        return current

    dx, dy = current.x - parent.x, current.y - parent.y
    if dx != 0 and dy != 0:
        if (
            not walkable(grid, current.x - dx, current.y)
            and walkable(grid, current.x - dx, current.y + dy)
        ) or (
            not walkable(grid, current.x, current.y - dy)
            and walkable(grid, current.x + dx, current.y - dy)
        ):
            return current

        if jump(grid, Point(current.x + dx, current.y), current, end) or jump(
            grid, Point(current.x, current.y + dy), current, end
        ):
            return current
    elif dx != 0:
        if (
            not walkable(grid, current.x, current.y - 1)
            and walkable(grid, current.x + dx, current.y - 1)
        ) or (
            not walkable(grid, current.x, current.y + 1)
            and walkable(grid, current.x + dx, current.y + 1)
        ):
            return current
    else:
        if (
            not walkable(grid, current.x - 1, current.y)
            and walkable(grid, current.x - 1, current.y + dy)
        ) or (
            not walkable(grid, current.x + 1, current.y)
            and walkable(grid, current.x + 1, current.y + dy)
        ):
            return current

    if walkable(grid, current.x + dx, current.y) or walkable(
        grid, current.x, current.y + dy
    ):
        return jump(grid, Point(current.x + dx, current.y + dy), current, end)


def prune_neighbours(current: Point, parent: Point) -> Generator[Point, None, None]:
    dx, dy = (
        (current.x - parent.x) // max(abs(current.x - parent.x), 1),
        (current.y - parent.y) // max(abs(current.y - parent.y), 1),
    )

    if dx != 0 and dy != 0:
        yield Point(current.x + dx, current.y)
        yield Point(current.x, current.y + dy)
        yield Point(current.x + dx, current.y + dy)

        # not sure about these
        yield Point(current.x + dx, current.y - dy)
        yield Point(current.x - dx, current.y + dy)
    elif dx != 0:
        yield Point(current.x + dx, current.y - 1)
        yield Point(current.x + dx, current.y)
        yield Point(current.x + dx, current.y + 1)
    elif dy != 0:
        yield Point(current.x - 1, current.y + dy)
        yield Point(current.x, current.y + dy)
        yield Point(current.x + 1, current.y + dy)
    else:
        yield from (
            Point(current.x + dx, current.y + dy) for dx, dy in INTERCARDINAL_OFFSETS
        )


def calculate_astar_path(grid: np.ndarray, start: Point, end: Point) -> list[Point]:
    heap: list[tuple[int, Point, Point]] = [(0, start, start)]
    came_from: dict[Point, Point] = {start: start}
    distances: dict[Point, int] = {start: 0}

    while heap:
        _, current, parent = heappop(heap)

        if current == end:
            result = []
            while True:
                result.append(current)

                if came_from[current] != current:
                    current = came_from[current]
                else:
                    break
            return result

        for neighbour in prune_neighbours(current, parent):
            jump_point = jump(grid, neighbour, current, end)
            if jump_point and jump_point not in came_from:
                came_from[jump_point] = current
                distances[jump_point] = distances[came_from[jump_point]] + 1
                f_cost = distances[jump_point] + max(
                    abs(jump_point.x - current.x), abs(jump_point.y - current.y)
                )

                heappush(heap, (f_cost, jump_point, current))

    return []
