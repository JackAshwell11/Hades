"""Creates a binary space partition which is used for procedurally generating the game
map."""
from __future__ import annotations

# Builtin
import random
from typing import NamedTuple

# Pip
import numpy as np

# Custom
from game.constants.generation import (
    HALLWAY_SIZE,
    MIN_CONTAINER_SIZE,
    MIN_ROOM_SIZE,
    TileType,
)

__all__ = (
    "Point",
    "Rect",
    "Leaf",
)


class Point(NamedTuple):
    """Represents a point in the grid.

    Parameters
    ----------
    x: int
        The x position.
    y: int
        The y position.
    """

    x: int
    y: int

    def __repr__(self) -> str:
        return f"<Point (X={self.x}) (Y={self.y})>"


class Rect:
    """Represents a rectangle of any size useful for creating the dungeon. Containers
    include the split wall in their sizes whereas rooms don't so MIN_CONTAINER_SIZE must
    be bigger than MIN_ROOM_SIZE.

    Parameters
    ----------
    top_left: Point
        The top-left position.
    bottom_right: Point
        The bottom-right position
    """

    __slots__ = (
        "top_left",
        "bottom_right",
    )

    def __init__(self, top_left: Point, bottom_right: Point) -> None:
        self.top_left: Point = top_left
        self.bottom_right: Point = bottom_right

    def __repr__(self) -> str:
        return f"<Rect (Top-left={self.top_left}) (Bottom-right={self.bottom_right})>"

    @property
    def width(self) -> int:
        """Gets the width of the rect.

        Returns
        -------
        int
            The width of the rect.
        """
        return abs(self.bottom_right.x - self.top_left.x)

    @property
    def height(self) -> int:
        """Gets the height of the rect.

        Returns
        -------
        int
            The height of the rect.
        """
        return abs(self.bottom_right.y - self.top_left.y)

    @property
    def center_x(self) -> int:
        """Gets the x coordinate of the center position.

        Returns
        -------
        int
            The x coordinate of the center position.
        """
        return int((self.top_left.x + self.bottom_right.x) / 2)

    @property
    def center_y(self) -> int:
        """Gets the y coordinate of the center position.

        Returns
        -------
        int
            The y coordinate of the center position.
        """
        return int((self.top_left.y + self.bottom_right.y) / 2)


class Leaf:
    """A binary spaced partition leaf which can be used to generate a dungeon.

    Parameters
    ----------
    top_left: Point
        The top-left position.
    bottom_right: Point
        The bottom-right position
    parent: Leaf | None
        The parent leaf object.
    grid: np.ndarray
        The 2D grid which represents the dungeon.

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
        Whether or not the leaf was split vertically. By default, this is None
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
        parent: Leaf | None,
        grid: np.ndarray,
    ) -> None:
        self.left: Leaf | None = None
        self.right: Leaf | None = None
        self.parent: Leaf | None = parent
        self.grid: np.ndarray = grid
        self.container: Rect = Rect(top_left, bottom_right)
        self.room: Rect | None = None
        self.split_vertical: bool | None = None

    def __repr__(self) -> str:
        return (
            f"<Leaf (Left={self.left}) (Right={self.right}) (Top-left"
            f" position={self.container.top_left}) (Bottom-right"
            f" position={self.container.bottom_right})>"
        )

    def split(self, debug_lines: bool = False) -> bool:
        """Splits a container horizontally or vertically.

        Parameters
        ----------
        debug_lines: bool
            Whether or not to draw the debug lines.

        Returns
        -------
        bool
            Whether the split was successful or not.
        """
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
            pos = self.container.top_left.x + pos
            if debug_lines:
                self.grid[
                    self.container.top_left.y : self.container.bottom_right.y + 1, pos
                ] = TileType.DEBUG_WALL.value

            # Create child leafs
            self.left = Leaf(
                Point(self.container.top_left.x, self.container.top_left.y),
                Point(pos - 1, self.container.bottom_right.y),
                self,
                self.grid,
            )
            self.right = Leaf(
                Point(pos + 1, self.container.top_left.y),
                Point(self.container.bottom_right.x, self.container.bottom_right.y),
                self,
                self.grid,
            )
        else:
            # Split horizontally making sure to adjust pos, so it can be within range of
            # the actual container
            pos = self.container.top_left.y + pos
            if debug_lines:
                self.grid[
                    pos, self.container.top_left.x : self.container.bottom_right.x + 1
                ] = TileType.DEBUG_WALL.value

            # Create child leafs
            self.left = Leaf(
                Point(self.container.top_left.x, self.container.top_left.y),
                Point(self.container.bottom_right.x, pos - 1),
                self,
                self.grid,
            )
            self.right = Leaf(
                Point(self.container.top_left.x, pos + 1),
                Point(self.container.bottom_right.x, self.container.bottom_right.y),
                self,
                self.grid,
            )

        # Set the leaf's split direction
        self.split_vertical = split_vertical

        # Successful split
        return True

    def create_room(self) -> bool:
        """Creates a random sized room inside a container.

        Returns
        -------
        bool
            Whether the room creation was successful or not.
        """
        # Test if this container is already split or not. If it is, we do not want to
        # create a room inside it otherwise it will overwrite other rooms
        if self.left is not None and self.right is not None:
            return False

        # Pick a random width and height making sure it is at least MIN_ROOM_SIZE but
        # doesn't exceed the container
        width = random.randint(MIN_ROOM_SIZE, self.container.width)
        height = random.randint(MIN_ROOM_SIZE, self.container.height)

        # Use the width and height to find a suitable x and y position which can create
        # the room
        x_pos = random.randint(
            self.container.top_left.x, self.container.bottom_right.x - width
        )
        y_pos = random.randint(
            self.container.top_left.y, self.container.bottom_right.y - height
        )

        # Create the room rect
        self.room = Rect(
            Point(x_pos, y_pos), Point(x_pos + width - 1, y_pos + height - 1)
        )

        # Place the room rect in the 2D grid
        self.place_rect(self.room)

        # Successful room creation
        return True

    def create_hallway(self, target: Leaf) -> tuple[Rect | None, Rect | None]:
        """Creates the hallway links between rooms.

        Parameters
        ----------
        target: Leaf
            The target leaf to make a hallway too.

        Returns
        -------
        tuple[Rect | None, Rect | None]
            A tuple containing the generated hallways. This may only contain 1 hallway
            since the other hallway may be contained entirely in the target room.
        """
        # Get the two rooms which will be connected with a hallway and make sure they're
        # valid
        start_room = self.room
        target_room = target.room
        assert start_room is not None
        assert target_room is not None

        # Get the split_vertical bool from the parent object, so we know which
        # orientation the hallway should be
        assert self.parent is not None
        split_vertical = self.parent.split_vertical

        # Create hallway intersection point. This will be used to determine which
        # orientation the hallway is out of 8 orientations. We also have the is_left and
        # split_vertical bools to help determine the orientation. The matches are:
        #   RIGHT-UP: split_vertical=True
        #   RIGHT-DOWN: split_vertical=True
        #   LEFT-UP: split_vertical=True
        #   LEFT-DOWN: split_vertical=True
        #   UP-LEFT: split_vertical=False
        #   UP-RIGHT: split_vertical=False
        #   DOWN-LEFT: split_vertical=False
        #   DOWN-RIGHT: split_vertical=False
        if split_vertical:
            hallway_intersection_x, hallway_intersection_y = (
                target_room.center_x,
                start_room.center_y,
            )
        else:
            hallway_intersection_x, hallway_intersection_y = (
                start_room.center_x,
                target_room.center_y,
            )

        # Determine hallway width/height
        half_hallway_size = HALLWAY_SIZE // 2

        # Set the base variables for the hallways
        first_top_left = [
            start_room.center_x - half_hallway_size,
            start_room.center_y - half_hallway_size,
        ]
        first_bottom_right = [
            start_room.center_x + half_hallway_size,
            start_room.center_y + half_hallway_size,
        ]
        second_top_left = [
            target_room.center_x - half_hallway_size,
            target_room.center_y - half_hallway_size,
        ]
        second_bottom_right = [
            target_room.center_x + half_hallway_size,
            target_room.center_y + half_hallway_size,
        ]

        # Determine if we need to change the first hallway's points based on its
        # orientation (or if we even need one at al)
        first_hallway_valid = False
        if hallway_intersection_x > start_room.bottom_right.x and split_vertical:
            # First hallway is right
            first_top_left[0] = start_room.bottom_right.x - 1
            first_bottom_right[0] = hallway_intersection_x + half_hallway_size + 1
            first_hallway_valid = True
        elif hallway_intersection_y > start_room.bottom_right.y and not split_vertical:
            # First hallway is down
            first_top_left[1] = start_room.bottom_right.y - 1
            first_bottom_right[1] = hallway_intersection_y + half_hallway_size + 1
            first_hallway_valid = True
        elif hallway_intersection_x < start_room.top_left.x and split_vertical:
            # First hallway is left
            first_top_left[0] = hallway_intersection_x - half_hallway_size - 1
            first_bottom_right[0] = start_room.top_left.x + 1
            first_hallway_valid = True
        elif hallway_intersection_y < start_room.top_left.y and not split_vertical:
            # First hallway is up
            first_top_left[1] = hallway_intersection_y - half_hallway_size - 1
            first_bottom_right[1] = start_room.top_left.y + 1
            first_hallway_valid = True

        # Determine if we need to change the second hallway's points based on its
        # orientation (or if we even need one at al)
        second_hallway_valid = False
        if hallway_intersection_x < target_room.top_left.x and not split_vertical:
            # Second hallway is right
            second_top_left[0] = hallway_intersection_x - half_hallway_size - 1
            second_bottom_right[0] = target_room.top_left.x + 1
            second_hallway_valid = True
        elif hallway_intersection_y < target_room.top_left.y and split_vertical:
            # Second hallway is down
            second_top_left[1] = hallway_intersection_y - half_hallway_size - 1
            second_bottom_right[1] = target_room.top_left.y + 1
            second_hallway_valid = True
        elif hallway_intersection_x > target_room.bottom_right.x and not split_vertical:
            # Second hallway is left
            second_top_left[0] = target_room.bottom_right.x - 1
            second_bottom_right[0] = hallway_intersection_x + half_hallway_size + 1
            second_hallway_valid = True
        elif hallway_intersection_y > target_room.bottom_right.y and split_vertical:
            # Second hallway is up
            second_top_left[1] = target_room.bottom_right.y - 1
            second_bottom_right[1] = hallway_intersection_y + half_hallway_size + 1
            second_hallway_valid = True

        # Place the hallways
        first_hallway = None
        if first_hallway_valid:
            first_hallway = Rect(
                Point(*first_top_left),
                Point(*first_bottom_right),
            )
            self.place_rect(first_hallway)
        second_hallway = None
        if second_hallway_valid:
            second_hallway = Rect(
                Point(*second_top_left),
                Point(*second_bottom_right),
            )
            self.place_rect(second_hallway)

        # Return both hallways
        return first_hallway, second_hallway

    def place_rect(self, rect: Rect) -> None:
        """Places a rect in the 2D grid.

        Parameters
        ----------
        rect: Rect
            The rect to place in the 2D grid.
        """
        # Get the width and height of the array
        height, width = self.grid.shape

        # Place the walls
        temp = self.grid[
            max(rect.top_left.y, 0) : min(rect.bottom_right.y + 1, height),
            max(rect.top_left.x, 0) : min(rect.bottom_right.x + 1, width),
        ]
        temp[temp == TileType.EMPTY.value] = TileType.WALL.value

        # Place the floors. The ranges must be -1 in all directions since we don't want
        # to overwrite the walls keeping the player in, but we still want to overwrite
        # walls that block the path for hallways
        self.grid[
            max(rect.top_left.y + 1, 1) : min(rect.bottom_right.y, height - 1),
            max(rect.top_left.x + 1, 1) : min(rect.bottom_right.x, width - 1),
        ] = TileType.FLOOR.value
