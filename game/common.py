"""Holds common functionality that is shared between modules and files."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ("grid_bfs",)


cardinal_offsets: tuple[tuple[int, int], ...] = (
    (0, -1),
    (-1, 0),
    (1, 0),
    (0, 1),
)


def grid_bfs(
    target: tuple[int, int],
    height: int,
    width: int,
    offsets: tuple[tuple[int, int], ...] = cardinal_offsets,
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
    offsets: tuple[tuple[int, int], ...]
        A tuple of offsets used for getting the target's neighbours.

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
        yield x, y
