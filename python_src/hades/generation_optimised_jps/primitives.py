"""Stores objects that are shared between all generation classes."""
from __future__ import annotations

# Builtin
from typing import NamedTuple

# Pip
import numpy as np
import numpy.typing as npt

# Custom
from hades.constants.generation import REPLACEABLE_TILES, TileType

__all__ = (
    "Point",
    "Rect",
)


class Point(NamedTuple):
    """Represents a point in the grid.

    x: int
        The x position.
    y: int
        The y position.
    """

    x: int
    y: int

    def point_sum(self, other: Point) -> tuple[int, int]:
        return self.x + other.x, self.y + other.y

    def abs_diff(self, other: Point) -> tuple[int, int]:
        return abs(self.x - other.x), abs(self.y - other.y)


class Rect(NamedTuple):
    """Represents a rectangle of any size useful for the interacting with the 2D grid.

    When creating a container, the split wall is included in the rect size, whereas,
    rooms don't so MIN_CONTAINER_SIZE must be bigger than MIN_ROOM_SIZE.

    top_left: Point
        The top-left position.
    bottom_right: Point
        The bottom-right position.
    """

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        """Get the width of the rect.

        Returns
        -------
        int
            The width of the rect.
        """
        return self.bottom_right.abs_diff(self.top_left)[0]

    @property
    def height(self) -> int:
        """Get the height of the rect.

        Returns
        -------
        float
            The height of the rect.
        """
        return self.bottom_right.abs_diff(self.top_left)[1]

    @property
    def center(self) -> Point:
        """Get the center position of the rect.

        Returns
        -------
        Point
            The center position of the rect.
        """
        sum_pnt = self.top_left.point_sum(self.bottom_right)
        return Point(round(sum_pnt[0] / 2), round(sum_pnt[1] / 2))

    def get_distance_to(self, other: Rect) -> float:
        """Get the Euclidean distance to another rect.

        Parameters
        ----------
        other: Rect
            The rect to find the distance to.

        Returns
        -------
        float
            The Euclidean distance between this rect and the given rect.
        """
        return max(
            abs(self.center.x - other.center.x), abs(self.center.y - other.center.y)
        )

    def place_rect(self, grid: npt.NDArray[np.int8]) -> None:
        """Places the rect in the 2D grid.

        Parameters
        ----------
        grid: npt.NDArray[np.int8]
            The 2D grid which represents the dungeon.
        """
        # Get the width and height of the grid
        grid_height, grid_width = grid.shape

        # Place the walls
        temp_wall = grid[
            max(self.top_left.y, 0) : min(self.bottom_right.y + 1, grid_height),
            max(self.top_left.x, 0) : min(self.bottom_right.x + 1, grid_width),
        ]
        temp_wall[np.isin(temp_wall, REPLACEABLE_TILES)] = TileType.WALL

        # Place the floors. The ranges must be -1 in all directions since we don't want
        # to overwrite the walls keeping the player in, but we still want to overwrite
        # walls that block the path for hallways
        grid[
            max(self.top_left.y + 1, 1) : min(self.bottom_right.y, grid_height - 1),
            max(self.top_left.x + 1, 1) : min(self.bottom_right.x, grid_width - 1),
        ] = TileType.FLOOR

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return (
            f"<Rect (Top left position={self.top_left}) (Bottom right"
            f" position={self.bottom_right}) (Center position={self.center})"
            f" (Width={self.width}) (Height={self.height})>"
        )
