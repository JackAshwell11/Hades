"""Creates a binary space partition used for generating the rooms."""
from __future__ import annotations

# Builtin
import random
from typing import TYPE_CHECKING

# Custom
from hades.constants.general import DEBUG_GAME
from hades.constants.generation import (
    MIN_CONTAINER_SIZE,
    MIN_ROOM_SIZE,
    ROOM_RATIO,
    TileType,
)
from hades.generation.primitives import Point, Rect

if TYPE_CHECKING:
    import numpy as np

__all__ = ("Leaf",)


class Leaf:
    """A binary spaced partition leaf used to generate the dungeon's rooms..

    Parameters
    ----------
    top_left: Point
        The top-left position.
    bottom_right: Point
        The bottom-right position
    grid: np.ndarray
        The 2D grid which represents the dungeon.
    parent: Leaf | None
        The parent leaf object.

    Attributes
    ----------
    left: Leaf | None
        The left container of this leaf. If this is None, we have reached the end of the
        branch.
    right: Leaf | None
        The right container of this leaf. If this is None, we have reached the end of
        the branch.
    container: Rect
        The rect object for representing this leaf.
    room: Rect | None
        The rect object for representing the room inside this leaf.
    split_vertical: bool | None
        Whether the leaf was split vertically or not. By default, this is None
        (not split).
    """

    __slots__ = (
        "left",
        "right",
        "parent",
        "container",
        "room",
        "grid",
        "split_vertical",
    )

    def __init__(
        self,
        top_left: Point,
        bottom_right: Point,
        grid: np.ndarray,
        parent: Leaf | None = None,
    ) -> None:
        self.left: Leaf | None = None
        self.right: Leaf | None = None
        self.parent: Leaf | None = parent
        self.grid: np.ndarray = grid
        self.container: Rect = Rect(grid, top_left, bottom_right)
        self.room: Rect | None = None
        self.split_vertical: bool | None = None

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return (
            f"<Leaf (Left={self.left}) (Right={self.right}) (Top-left"
            f" position={self.container.top_left}) (Bottom-right"
            f" position={self.container.bottom_right})>"
        )

    def split(self) -> bool:
        """Split a container either horizontally or vertically.

        Returns
        -------
        bool
            Whether the split was successful or not.
        """
        # Check if this leaf is already split or not
        if self.left and self.right:
            return False

        # To determine the direction of split, we test if the width is 25% larger than
        # the height, if so we split vertically. However, if the height is 25% larger
        # than the width, we split horizontally. Otherwise, we split randomly
        split_vertical = bool(random.getrandbits(1))
        if (
            self.container.width > self.container.height
            and self.container.width / self.container.height >= 1.25
        ):
            split_vertical = True
        elif (
            self.container.height > self.container.width
            and self.container.height / self.container.width >= 1.25
        ):
            split_vertical = False

        # To determine the range of values that we could split on, we need to find out
        # if the container is too small. Once we've done that, we can use the x1, y1, x2
        # and y2 coordinates to specify the range of values
        max_size = (
            self.container.width - MIN_CONTAINER_SIZE
            if split_vertical
            else self.container.height - MIN_CONTAINER_SIZE
        )
        if max_size <= MIN_CONTAINER_SIZE:
            # Container too small to split
            return False

        # Create the split position. This ensures that there will be MIN_CONTAINER_SIZE
        # on each side
        pos = random.randint(MIN_CONTAINER_SIZE, max_size)

        # Split the container
        if split_vertical:
            # Split vertically making sure to adjust pos, so it can be within range of
            # the actual container
            pos += self.container.top_left.x
            if DEBUG_GAME:  # pragma: no branch
                self.grid[
                    self.container.top_left.y : self.container.bottom_right.y + 1, pos
                ] = TileType.DEBUG_WALL

            # Create child leafs
            self.left = Leaf(
                Point(self.container.top_left.x, self.container.top_left.y),
                Point(pos - 1, self.container.bottom_right.y),
                self.grid,
                self,
            )
            self.right = Leaf(
                Point(pos + 1, self.container.top_left.y),
                Point(self.container.bottom_right.x, self.container.bottom_right.y),
                self.grid,
                self,
            )
        else:
            # Split horizontally making sure to adjust pos, so it can be within range of
            # the actual container
            pos += self.container.top_left.y
            if DEBUG_GAME:  # pragma: no branch
                self.grid[
                    pos, self.container.top_left.x : self.container.bottom_right.x + 1
                ] = TileType.DEBUG_WALL

            # Create child leafs
            self.left = Leaf(
                Point(self.container.top_left.x, self.container.top_left.y),
                Point(self.container.bottom_right.x, pos - 1),
                self.grid,
                self,
            )
            self.right = Leaf(
                Point(self.container.top_left.x, pos + 1),
                Point(self.container.bottom_right.x, self.container.bottom_right.y),
                self.grid,
                self,
            )

        # Set the leaf's split direction
        self.split_vertical = split_vertical

        # Successful split
        return True

    def create_room(self) -> bool:
        """Create a random sized room inside a container.

        Returns
        -------
        bool
            Whether the room creation was successful or not.
        """
        # Test if this container is already split or not. If it is, we do not want to
        # create a room inside it otherwise it will overwrite other rooms
        if self.left and self.right:
            return False

        # Pick a random width and height making sure it is at least MIN_ROOM_SIZE but
        # doesn't exceed the container
        width = random.randint(MIN_ROOM_SIZE, int(self.container.width))
        height = random.randint(MIN_ROOM_SIZE, int(self.container.height))

        # Use the width and height to find a suitable x and y position which can create
        # the room
        x_pos = random.randint(
            int(self.container.top_left.x), int(self.container.bottom_right.x - width)
        )
        y_pos = random.randint(
            int(self.container.top_left.y), int(self.container.bottom_right.y - height)
        )

        # Create the room rect and test if its width to height ratio will make an
        # oddly-shaped room
        rect = Rect(
            self.grid, Point(x_pos, y_pos), Point(x_pos + width - 1, y_pos + height - 1)
        )
        if (min(rect.width, rect.height) / max(rect.width, rect.height)) < ROOM_RATIO:
            return False

        # Width to height ratio is fine so store the rect and place it in the 2D grid
        self.room = rect
        self.room.place_rect()

        # Successful room creation
        return True
