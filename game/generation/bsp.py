from __future__ import annotations

# Builtin
import random

# Pip
import numpy as np

# Custom
from constants import HALLWAY_SIZE, MIN_CONTAINER_SIZE, MIN_ROOM_SIZE, TileType


class Rect:
    """
    Represents a rectangle of any size useful for creating the dungeon. Containers
    include the split wall in their sizes whereas rooms don't so MIN_CONTAINER_SIZE must
    be bigger than MIN_ROOM_SIZE.

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
        return abs(self.x2 - self.x1 + 1)

    @property
    def height(self) -> int:
        """Returns the height of the rect."""
        return abs(self.y2 - self.y1 + 1)

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

    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        grid: np.ndarray,
    ) -> None:
        self.left: Leaf | None = None
        self.right: Leaf | None = None
        self.container: Rect = Rect(x1, y1, x2, y2)
        self.room: Rect | None = None
        self.grid: np.ndarray = grid
        self.split_vertical: bool | None = None

    def __repr__(self) -> str:
        return (
            f"<Leaf (Left={self.left}) (Right={self.right}) (Top-left"
            f" position=({self.container.x1}, {self.container.y1})) (Bottom-right"
            f" position=({self.container.x2}, {self.container.y2}))>"
        )

    def split(self, debug_lines: bool = False) -> None:
        """
        Splits a container either horizontally or vertically.

        Parameters
        ----------
        debug_lines: bool
            Whether or not to draw the debug lines.
        """
        # Test if this container is already split or not. This container will always
        # have a left and right leaf when splitting since the checks later on in this
        # function ensure it is big enough to be split
        if self.left is not None and self.right is not None:
            self.left.split(debug_lines)
            self.right.split(debug_lines)
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
            if debug_lines:
                self.grid[
                    self.container.y1 : self.container.y2 + 1, pos
                ] = TileType.DEBUG_WALL.value

            # Create child leafs
            self.left = Leaf(
                self.container.x1,
                self.container.y1,
                pos - 1,
                self.container.y2,
                self.grid,
            )
            self.right = Leaf(
                pos + 1,
                self.container.y1,
                self.container.x2,
                self.container.y2,
                self.grid,
            )
        else:
            # Split horizontally making sure to adjust pos, so it can be within range of
            # the actual container
            pos = self.container.y1 + pos
            if debug_lines:
                self.grid[
                    pos, self.container.x1 : self.container.x2 + 1
                ] = TileType.DEBUG_WALL.value

            # Create child leafs
            self.left = Leaf(
                self.container.x1,
                self.container.y1,
                self.container.x2,
                pos - 1,
                self.grid,
            )
            self.right = Leaf(
                self.container.x1,
                pos + 1,
                self.container.x2,
                self.container.y2,
                self.grid,
            )

        # Set the leaf's split direction
        self.split_vertical = split_vertical

    def create_room(self, rooms: list[Rect]) -> None:
        """
        Creates a random sized room inside a container.

        Parameters
        ----------
        rooms: list[Rect]
            A list of all the generated rooms.
        """
        # Test if this container is already split or not. If it is, we do not want to
        # create a room inside it otherwise it will overwrite other rooms
        if self.left is not None and self.right is not None:
            self.left.create_room(rooms)
            self.right.create_room(rooms)
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

        # Create the room rect
        self.room = Rect(x_pos, y_pos, x_pos + width - 1, y_pos + height - 1)

        # Place the room rect in the 2D grid
        rooms.append(self.room)
        self.place_rect(self.room)

    def create_hallway(
        self,
        hallways: list[Rect],
        left_room: Rect | None = None,
        right_room: Rect | None = None,
    ) -> Rect:
        """
        Creates the hallway links between rooms. This uses a post-order traversal
        since we want to work our way up from the bottom of the tree.

        To save the hallways, we are using the fact that mutable data structures in
        Python don't create new objects when passed as variables therefore as long as we
        pass the list on each call, adding new items to the list will modify the
        original list simplifying the code.

        Parameters
        ----------
        hallways: list[Rect]
            A list of all the generated hallways.
        left_room: Rect | None
            The left room to create a hallway too.
        right_room: Rect | None
            The right room to create a hallway too.

        Returns
        -------
        Rect
            A room to create a hallway too.
        """
        # Traverse the left tree to connect its rooms
        if self.left is not None:
            left_room = self.left.create_hallway(hallways)

        # Traverse the right tree to connect its rooms
        if self.right is not None:
            right_room = self.right.create_hallway(hallways)

        # Return the current's leaf's room, so it can be connected with a matching one
        if self.room is not None:
            return self.room

        # Make sure that the rooms are not None
        assert left_room is not None and right_room is not None

        # Create hallway intersection point. This will be used to determine which
        # orientation the hallway is out of 8 orientations which are: RIGHT-UP,
        # RIGHT-DOWN, LEFT-UP, LEFT-DOWN, UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT
        if self.split_vertical:
            hallway_intersection_x, hallway_intersection_y = (
                right_room.center_x,
                left_room.center_y,
            )
        else:
            hallway_intersection_x, hallway_intersection_y = (
                left_room.center_x,
                right_room.center_y,
            )

        # Determine hallway width/height
        half_hallway_size = HALLWAY_SIZE // 2

        # Set the base variables for the hallways
        first_top_left = [
            left_room.center_x - half_hallway_size,
            left_room.center_y - half_hallway_size,
        ]
        first_bottom_right = [
            left_room.center_x + half_hallway_size,
            left_room.center_y + half_hallway_size,
        ]
        second_top_left = [
            right_room.center_x - half_hallway_size,
            right_room.center_y - half_hallway_size,
        ]
        second_bottom_right = [
            right_room.center_x + half_hallway_size,
            right_room.center_y + half_hallway_size,
        ]

        # Determine if we need to change the first hallway's points based on its
        # orientation
        if hallway_intersection_x >= left_room.center_x and self.split_vertical:
            # First hallway is right
            first_top_left[0] = left_room.x2 - 1
            first_bottom_right[0] = hallway_intersection_x + half_hallway_size + 1
        elif hallway_intersection_y >= left_room.center_y and not self.split_vertical:
            # First hallway is down
            first_top_left[1] = left_room.y2 - 1
            first_bottom_right[1] = hallway_intersection_y + half_hallway_size + 1
        elif hallway_intersection_x <= left_room.center_x and self.split_vertical:
            # First hallway is left
            first_top_left[0] = hallway_intersection_x - half_hallway_size - 1
            first_bottom_right[0] = left_room.x1 + 1
        elif hallway_intersection_y <= left_room.center_y and not self.split_vertical:
            # First hallway is up
            first_top_left[1] = hallway_intersection_y - half_hallway_size - 1
            first_bottom_right[1] = left_room.y1 + 1

        # Determine if we need to change the second hallway's points based on its
        # orientation (or if we even need one at al)
        valid = False
        if (
            hallway_intersection_x <= right_room.center_x
            and not self.split_vertical
            and hallway_intersection_x < right_room.x1
        ):
            # Second hallway is right
            second_top_left[0] = hallway_intersection_x - half_hallway_size
            second_bottom_right[0] = right_room.x1 + 1
            valid = True
        elif (
            hallway_intersection_y <= right_room.center_y
            and self.split_vertical
            and hallway_intersection_y < right_room.y1
        ):
            # Second hallway is down
            second_top_left[1] = hallway_intersection_y - half_hallway_size
            second_bottom_right[1] = right_room.y1 + 1
            valid = True
        elif (
            hallway_intersection_x >= right_room.center_x
            and not self.split_vertical
            and hallway_intersection_x > right_room.x2
        ):
            # Second hallway is left
            second_top_left[0] = right_room.x2 - 1
            second_bottom_right[0] = hallway_intersection_x + half_hallway_size
            valid = True
        elif (
            hallway_intersection_y >= right_room.center_y
            and self.split_vertical
            and hallway_intersection_y > right_room.y2
        ):
            # Second hallway is up
            second_top_left[1] = right_room.y2 - 1
            second_bottom_right[1] = hallway_intersection_y + half_hallway_size
            valid = True

        # Place the hallways
        first_hallway = Rect(
            *first_top_left,
            *first_bottom_right,
        )
        hallways.append(first_hallway)
        self.place_rect(first_hallway)
        if valid:
            second_hallway = Rect(
                *second_top_left,
                *second_bottom_right,
            )
            hallways.append(second_hallway)
            self.place_rect(second_hallway)

        # Return a room, so it can be connected on the next level
        if self.split_vertical:
            return right_room
        return left_room if bool(random.getrandbits(1)) else right_room

    def place_rect(self, rect: Rect) -> None:
        """
        Places a rect in the 2D grid.

        Parameters
        ----------
        rect: Rect
            The rect to place in the 2D grid.
        """
        # Get the width and height of the array
        height, width = self.grid.shape

        # Place the walls
        temp = self.grid[
            rect.y1 : min(rect.y2 + 1, height),
            rect.x1 : min(rect.x2 + 1, width),
        ]
        temp[temp == TileType.EMPTY.value] = TileType.WALL.value

        # Place the floors. The ranges must be -1 in all directions since we don't want
        # to overwrite the walls keeping the player in, but we still want to overwrite
        # walls that block the path for hallways
        self.grid[
            rect.y1 + 1 : min(rect.y2, height - 1),
            rect.x1 + 1 : min(rect.x2, width - 1),
        ] = TileType.FLOOR.value
