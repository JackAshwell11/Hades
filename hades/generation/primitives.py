"""Stores objects that are shared between all generation classes."""
from __future__ import annotations

# Builtin
from typing import NamedTuple

# Pip
import numpy as np

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


class Rect(NamedTuple):
    """Represents a rectangle of any size useful for creating the dungeon.

    Containers include the split wall in their sizes whereas rooms don't so
    MIN_CONTAINER_SIZE must be bigger than MIN_ROOM_SIZE.

    grid: np.ndarray
        The 2D grid which represents the dungeon.
    top_left: Point
        The top-left position.
    bottom_right: Point
        The bottom-right position
    """

    grid: np.ndarray
    top_left: Point
    bottom_right: Point

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return (
            f"<Rect (Top left position={self.top_left}) (Bottom right"
            f" position={self.bottom_right}) (Center position={self.center})"
            f" (Width={self.width}) (Height={self.height})>"
        )

    @property
    def width(self) -> int:
        """Get the width of the rect.

        Returns
        -------
        float
            The width of the rect.
        """
        return abs(self.bottom_right.x - self.top_left.x)

    @property
    def height(self) -> int:
        """Get the height of the rect.

        Returns
        -------
        float
            The height of the rect.
        """
        return abs(self.bottom_right.y - self.top_left.y)

    @property
    def center_x(self) -> int:
        """Get the x coordinate of the center position.

        Returns
        -------
        int
            The x coordinate of the center position.
        """
        return round((self.top_left.x + self.bottom_right.x) / 2)

    @property
    def center_y(self) -> int:
        """Get the y coordinate of the center position.

        Returns
        -------
        int
            The y coordinate of the center position.
        """
        return round((self.top_left.y + self.bottom_right.y) / 2)

    @property
    def center(self) -> Point:
        """Get the center position of the rect.

        Returns
        -------
        Point
            The center position of the rect.
        """
        return Point(self.center_x, self.center_y)

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
        return np.hypot(other.center_x - self.center_x, other.center_y - self.center_y)

    def place_rect(self) -> None:
        """Places the rect in the 2D grid."""
        # Get the width and height of the grid
        grid_height, grid_width = self.grid.shape

        # Place the walls
        temp_wall = self.grid[
            max(self.top_left.y, 0) : min(self.bottom_right.y + 1, grid_height),
            max(self.top_left.x, 0) : min(self.bottom_right.x + 1, grid_width),
        ]
        temp_wall[np.isin(temp_wall, REPLACEABLE_TILES)] = TileType.WALL

        # Place the floors. The ranges must be -1 in all directions since we don't want
        # to overwrite the walls keeping the player in, but we still want to overwrite
        # walls that block the path for hallways
        self.grid[
            max(self.top_left.y + 1, 1) : min(self.bottom_right.y, grid_height - 1),
            max(self.top_left.x + 1, 1) : min(self.bottom_right.x, grid_width - 1),
        ] = TileType.FLOOR
