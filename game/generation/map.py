from __future__ import annotations

# Builtin
import random
from enum import Enum
from typing import List, Optional, Tuple

# Pip
import numpy as np

# Constants
EMPTY = 0
FLOOR = 1
WALL = 2
PLAYER_START = 3

MAX_WIDTH = 100
MAX_HEIGHT = 60
MAX_RECT_COUNT = 20


class Direction(Enum):
    """Represents a 4-point compass useful for determining where doors are."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Door:
    """
    Represents a door with a specific direction useful for spawning new rects.

    Parameters
    ----------
    position: Tuple[int, int]
        The position of the center point of the door. This is in the format (x, y).
    direction: Direction
        The direction the door is facing.
    rect: Rect
        The rectangle object that this door belongs too.
    """

    def __init__(
        self, position: Tuple[int, int], direction: Direction, rect: Rect
    ) -> None:
        self.position: Tuple[int, int] = position
        self.direction: Direction = direction
        self.rect: Rect = rect

    def __repr__(self) -> str:
        return f"<Door (Position={self.position}) (Direction={self.direction.name})>"


class Rect:
    """
    Represents a rectangle in the game of any size.

    Parameters
    ----------
    x: int
        The x position of the top-left corner.
    y: int
        The y position of the top-left corner
    width: int
        The width of the rectangle.
    height: int
        The height of the rectangle.
    parent: Optional[Rect]
        The rect object that spawned this current object.
    """

    def __init__(
        self, x: int, y: int, width: int, height: int, parent: Optional[Rect]
    ) -> None:
        self.x: int = x
        self.y: int = y
        self.width: int = width
        self.height: int = height
        self.parent: Optional[Rect] = parent

    def __repr__(self) -> str:
        return (
            f"<Rect (Position=({self.x}, {self.y})) (Width={self.width})"
            f" (Height={self.height}) (Parent={self.parent})>"
        )


class Map:
    """
    Procedurally generates a game generation based on a given game level.

    Parameters
    ----------
    width: int
        The width of the game generation.
    height: int
        The height of the game generation.

    Attributes
    ----------
    grid: np.ndarray
        The actual 2D matrix which represents the map.
    doors: List[Door] = []
        A list which holds all the locations of the doors.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width: int = width
        self.height: int = height
        self.grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self.doors: List[Door] = []

    def __repr__(self) -> str:
        return f"<Map (Width={self.width}) (Height={self.height})>"

    def make_map(self, level: int):
        """
        Function which actually does the map generation for a specified level.

        Parameters
        ----------
        level: int
            The level to create a game generation for.
        """

        # Create starting room
        self.make_start_room()
        # Generate rects
        self.generate_rects()

        print(self.grid)

    def make_start_room(self) -> None:
        """
        Creates a 5x5 starting room at a random position making sure it doesn't exceed
        the array bounds. Note that the int after the : is 1 bigger since numpy doesn't
        include the starting pos.

        This will create a 2D matrix at a random point that looks like:
        2  2  1  1  1  2  2
        2  1  1  1  1  1  2
        1  1  1  1  1  1  1
        1  1  1  3  1  1  1
        1  1  1  1  1  1  1
        2  1  1  1  1  1  2
        2  2  1  1  1  2  2
        """
        # Create starting position
        starting_pos_x, starting_pos_y = (
            random.randint(4, self.width - 4),
            random.randint(4, self.height - 4),
        )
        # Create walls
        self.grid[
            starting_pos_y - 3 : starting_pos_y + 4,
            starting_pos_x - 3 : starting_pos_x + 4,
        ] = WALL
        # Create inner floors
        self.grid[
            starting_pos_y - 2 : starting_pos_y + 3,
            starting_pos_x - 2 : starting_pos_x + 3,
        ] = FLOOR
        # Create door floors
        self.grid[
            starting_pos_y - 1 : starting_pos_y + 2,
            starting_pos_x - 3 : starting_pos_x + 4,
        ] = FLOOR
        self.grid[
            starting_pos_y - 3 : starting_pos_y + 4,
            starting_pos_x - 1 : starting_pos_x + 2,
        ] = FLOOR
        # Set player position
        self.grid[starting_pos_y, starting_pos_x] = PLAYER_START
        # Create starting rect object
        start_rect = Rect(starting_pos_x, starting_pos_y, 6, 6, None)
        # Update doors dict
        self.doors.append(
            Door((starting_pos_x, starting_pos_y - 3), Direction.NORTH, start_rect)
        )
        self.doors.append(
            Door((starting_pos_x + 3, starting_pos_y), Direction.EAST, start_rect)
        )
        self.doors.append(
            Door((starting_pos_x, starting_pos_y + 3), Direction.SOUTH, start_rect)
        )
        self.doors.append(
            Door((starting_pos_x - 3, starting_pos_y), Direction.WEST, start_rect)
        )

    def generate_rects(self) -> None:
        """"""
