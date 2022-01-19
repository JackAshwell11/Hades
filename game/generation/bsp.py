from __future__ import annotations

import random

# Builtin
from typing import Optional, Tuple

# Pip
import numpy as np

# Custom
from constants import DEBUG_WALL, EMPTY, FLOOR, MIN_CONTAINER_SIZE, MIN_ROOM_SIZE, WALL


class Rect:
    """
    Represents a rectangle of any size useful for creating the dungeon.

    Parameters
    ----------
    x1: int
        The top-left x position.
    y1: int
        The top-left y position.
    x2: int
        The bottom-right x position.
    y2: int
        The bottom-right y position.
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x2
        self.y2: int = y2

    def __repr__(self) -> str:
        return (
            f"<Rect (Top-left=({self.x1}, {self.y1})) (Bottom-right=({self.x2},"
            f" {self.y2}))>"
        )

    @property
    def width(self) -> int:
        """Returns the width of the rect."""
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        """Returns the height of the rect."""
        return self.y2 - self.y1 + 1

    @property
    def center_x(self) -> int:
        """Returns the x coordinate of the center position."""
        return int((self.x1 + self.x2) / 2)

    @property
    def center_y(self) -> int:
        """Returns the y coordinate of the center position."""
        return int((self.y1 + self.y2) / 2)


class Leaf:
    """
    A binary spaced partition leaf which can be used to generate a dungeon.

    Parameters
    ----------
    x1: int
        The top-left x position.
    y1: int
        The top-left y position.
    x2: int
        The bottom-right x position.
    y2: int
        The bottom-right y position.
    grid: np.ndarray
        The 2D grid which represents the dungeon.
    debug_lines: bool
        Whether or not to show the split lines. By default, this is False.

    Attributes
    ----------
    left: Optional[Leaf]
        The left container of this leaf. If this is None, we have reached the end of the
        branch.
    right: Optional[Leaf]
        The right container of this leaf. If this is None, we have reached the end of
        the branch.
    container: Rect
        The rect object for representing this leaf.
    room: Optional[Rect]
        The rect object for representing the room inside this leaf.
    split_vertical: Optional[bool]
        Whether or not the leaf was split vertically. By default, this is None
        (not split).
    """

    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        grid: np.ndarray,
        debug_lines: bool = False,
    ) -> None:
        self.left: Optional[Leaf] = None
        self.right: Optional[Leaf] = None
        self.container: Rect = Rect(x1, y1, x2, y2)
        self.room: Optional[Rect] = None
        self.grid: np.ndarray = grid
        self.split_vertical: Optional[bool] = None
        self.debug_lines: bool = debug_lines

    def __repr__(self) -> str:
        return (
            f"<Leaf (Left={self.left}) (Right={self.right})"
            f" (Top-left position=({self.container.x1}, {self.container.y1}))>"
        )

    def split(self) -> None:
        """Splits a container either horizontally or vertically."""
        # Test if this container is already split or not. This container will always
        # have a left and right leaf when splitting since the checks later on in this
        # function ensure it is big enough to be split
        if self.left is not None and self.right is not None:
            self.left.split()
            self.right.split()
            # Return just to make sure this container isn't split again with left and
            # right being overwritten
            return

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
        if split_vertical:
            max_size = self.container.width - MIN_CONTAINER_SIZE
        else:
            max_size = self.container.height - MIN_CONTAINER_SIZE
        if max_size <= MIN_CONTAINER_SIZE:
            # Container to small to split
            return

        # Create the split position. This ensures that there will be MIN_CONTAINER_SIZE
        # on each side
        pos = random.randint(MIN_CONTAINER_SIZE, max_size)

        # Split the container
        if split_vertical:
            # Split vertically making sure to adjust pos, so it can be within range of
            # the actual container
            pos = self.container.x1 + pos
            if self.debug_lines:
                self.grid[self.container.y1 : self.container.y2 + 1, pos] = DEBUG_WALL

            # Create child leafs
            self.left = Leaf(
                self.container.x1,
                self.container.y1,
                pos - 1,
                self.container.y2,
                self.grid,
                self.debug_lines,
            )
            self.right = Leaf(
                pos + 1,
                self.container.y1,
                self.container.x2,
                self.container.y2,
                self.grid,
                self.debug_lines,
            )
        else:
            # Split horizontally making sure to adjust pos, so it can be within range of
            # the actual container
            pos = self.container.y1 + pos
            if self.debug_lines:
                self.grid[pos, self.container.x1 : self.container.x2 + 1] = DEBUG_WALL

            # Create child leafs
            self.left = Leaf(
                self.container.x1,
                self.container.y1,
                self.container.x2,
                pos - 1,
                self.grid,
                self.debug_lines,
            )
            self.right = Leaf(
                self.container.x1,
                pos + 1,
                self.container.x2,
                self.container.y2,
                self.grid,
                self.debug_lines,
            )

        # Set the leaf's split direction
        self.split_vertical = split_vertical

    def create_room(self) -> None:
        """Creates a random sized room inside a container."""
        # Test if this container is already split or not. If it is, we do not want to
        # create a room inside it otherwise it will overwrite other rooms
        if self.left is not None and self.right is not None:
            self.left.create_room()
            self.right.create_room()
            # Return just to make sure this container doesn't create a room overwriting
            # others
            return

        # Pick a random width and height making sure it is at least MIN_ROOM_SIZE but
        # doesn't exceed the container
        width = random.randint(MIN_ROOM_SIZE, self.container.width)
        height = random.randint(MIN_ROOM_SIZE, self.container.height)
        # Use the width and height to find a suitable x and y position which can create
        # the room
        x_pos = random.randint(
            self.container.x1, self.container.x1 + self.container.width - width
        )
        y_pos = random.randint(
            self.container.y1, self.container.y1 + self.container.height - height
        )
        # Update the grid with the walls and floors
        self.grid[y_pos : y_pos + height, x_pos : x_pos + width] = WALL
        self.grid[y_pos + 1 : y_pos + height - 1, x_pos + 1 : x_pos + width - 1] = FLOOR
        # Create the room rect
        self.room = Rect(x_pos, y_pos, x_pos + width - 1, y_pos + height - 1)

    def create_hallway(
        self, left_room: Optional[Rect] = None, right_room: Optional[Rect] = None
    ) -> Rect:
        """
        Creates the hallway links between rooms. This uses a post-order traversal
        since we want to work our way up from the bottom of the tree.

        Parameters
        ----------
        left_room: Optional[Rect]
            The left room to create a hallway too.
        right_room: Optional[Rect]
            The right room to create a hallway too.

        Returns
        -------
        Rect
            A room to create a hallway too.
        """
        # Traverse the left tree to connect its rooms
        if self.left is not None:
            left_room = self.left.create_hallway()

        # Traverse the right tree to connect its rooms
        if self.right is not None:
            right_room = self.right.create_hallway()

        # Return the current's leaf's room, so it can be connected with a matching one
        if self.room is not None:
            return self.room

        # Make sure that the rooms are not None
        assert left_room is not None and right_room is not None

        # Connect the left and right rooms
        if self.split_vertical:
            # Leaf was split vertically so create a horizontal hallway
            self.horizontal_hallway(
                left_room.center_x, right_room.center_x, left_room.center_y
            )
            # Return the right room, so it can be connected on the next level
            return right_room
        else:
            # Leaf was split horizontally so create a vertical hallway
            self.vertical_hallway(
                left_room.center_y, right_room.center_y, left_room.center_x
            )
            # Return the left room, so it can be connected on the next level
            return left_room

    def horizontal_hallway(
        self, left_center_x: int, right_center_x: int, left_center_y: int
    ) -> None:
        """
        Creates a horizontal hallway from one point to another with a floor height of
        2.

        Parameters
        ----------
        left_center_x: int
            The x coordinate of the left room.

        right_center_x: int
            The x coordinate of the right room.

        left_center_y: int
            The y coordinate of the left room.
        """
        for x in range(left_center_x, right_center_x + 1):
            for y in range(left_center_y - 2, left_center_y + 2):
                if self.grid[y, x] == EMPTY:
                    self.grid[y, x] = WALL

        for x in range(left_center_x, right_center_x):
            for y in range(left_center_y - 1, left_center_y + 1):
                self.grid[y, x] = FLOOR

    def vertical_hallway(
        self, left_center_y: int, right_center_y: int, left_center_x: int
    ) -> None:
        """
        Creates a vertical hallway from one point to another with a floor width of 2.

        Parameters
        ----------
        left_center_y: int
            The y coordinate of the left room.

        right_center_y: int
            The y coordinate of the right room.

        left_center_x: int
            The x coordinate of the left room.
        """
        for y in range(left_center_y, right_center_y + 1):
            for x in range(left_center_x - 2, left_center_x + 2):
                if self.grid[y, x] == EMPTY:
                    self.grid[y, x] = WALL

        for y in range(left_center_y, right_center_y):
            for x in range(left_center_x - 1, left_center_x + 1):
                self.grid[y, x] = FLOOR

    def create_player_spawn(self) -> Tuple[int, int]:
        """Picks a random point which is a floor and creates the player spawn."""
        # We have to break this up otherwise mypy will return a Tuple[Any, ...] error
        player_spawn = random.choice(np.argwhere(self.grid == 1))
        return player_spawn[0], player_spawn[1]
