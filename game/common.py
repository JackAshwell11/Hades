"""Holds common functionality that is shared between modules and files."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from game.generation.primitives import Point

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ("grid_bfs",)


cardinal_offsets: list[tuple[int, int]] = [
    (0, -1),
    (-1, 0),
    (1, 0),
    (0, 1),
]


def grid_bfs(
    target: tuple[int, int],
    height: int,
    width: int,
    offsets: list[tuple[int, int]] = cardinal_offsets,
    return_point: bool = False,
) -> Generator[tuple[int, int], None, None]:
    """Get a target's neighbours based on a given list of offsets.

    Parameters
    ----------
    target: tuple[int, int]
        The target to get neighbours for.
    height: int
        The height of the grid.
    width: int
        The width of the grid.
    offsets: list[tuple[int, int]]
        A list of offsets used for getting them target's neighbours.
    return_point: bool
        Whether to return a Point object or not.

    Returns
    -------
    Generator[tuple[int, int], None, None]
        A list of the target's neighbours.
    """
    # Get all the neighbour floor tile positions relative to the current target
    for dx, dy in offsets:
        # Check if the neighbour position is within the boundaries of the grid or not
        x, y = target[0] + dx, target[1] + dy
        if (x < 0 or x >= width) or (y < 0 or y >= height):
            continue

        # Yield the neighbour tile position
        if return_point:
            yield Point(x, y)
        else:
            yield x, y
