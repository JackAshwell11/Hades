from __future__ import annotations

# Builtin
import random
from enum import Enum
from typing import Dict, List, Tuple

# Pip
import numpy as np

# Constants
EMPTY = 0
FLOOR = 1
WALL = 2
PLAYER_START = 3


class Direction(Enum):
    """Represents a 4-point compass useful for determining where doors are."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


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
    grid: List[List[int]]
        The actual 2D matrix which represents the map.
    doors: Dict[Tuple[int, int], Direction]
        A dict which holds all the locations of the doors.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width: int = width
        self.height: int = height
        self.grid: List[List[int]] = [
            [EMPTY for _ in range(self.width)] for _ in range(self.height)
        ]
        self.doors: Dict[Tuple[int, int], Direction] = {}

    def __repr__(self) -> str:
        return f"<Map (Width={self.width}) (Height={self.height})>"

    def make_map(self, level: int):
        """
        Function which actually creates the game generation for a specified level.

        Parameters
        ----------
        level: int
            The level to create a game generation for.
        """

        # Create starting room
        grid = np.array(self.grid)
        self.make_start_room(grid)

        import pandas as pd

        df = pd.DataFrame(grid)
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.width", 150)
        print(df)

    def make_start_room(self, grid: np.ndarray) -> None:
        """
        Creates a 5x5 starting room at a random position making sure it doesn't exceed
        the array bounds. Note that the int after the : is 1 bigger since numpy doesn't
        include the starting pos. This will create a 2D matrix at a random point that
        looks like:
        2  2  1  1  1  2  2
        2  1  1  1  1  1  2
        1  1  1  1  1  1  1
        1  1  1  3  1  1  1
        1  1  1  1  1  1  1
        2  1  1  1  1  1  2
        2  2  1  1  1  2  2

        Parameters
        ----------
        grid: np.ndarray
            The grid to create the starting room with. This is in-place.
        """
        # Create starting position
        starting_pos_x, starting_pos_y = (
            random.randint(4, self.width - 4),
            random.randint(4, self.height - 4),
        )
        # Create walls
        grid[
            starting_pos_y - 3 : starting_pos_y + 4,
            starting_pos_x - 3 : starting_pos_x + 4,
        ] = WALL
        # Create inner floors
        grid[
            starting_pos_y - 2 : starting_pos_y + 3,
            starting_pos_x - 2 : starting_pos_x + 3,
        ] = FLOOR
        # Create door floors
        grid[
            starting_pos_y - 1 : starting_pos_y + 2,
            starting_pos_x - 3 : starting_pos_x + 4,
        ] = FLOOR
        grid[
            starting_pos_y - 3 : starting_pos_y + 4,
            starting_pos_x - 1 : starting_pos_x + 2,
        ] = FLOOR
        # Set player position
        grid[starting_pos_y, starting_pos_x] = PLAYER_START
        # Update doors dict
        self.doors[(starting_pos_x, starting_pos_y - 3)] = Direction.NORTH
        self.doors[(starting_pos_x + 3, starting_pos_y)] = Direction.EAST
        self.doors[(starting_pos_x, starting_pos_y + 3)] = Direction.SOUTH
        self.doors[(starting_pos_x - 3, starting_pos_y)] = Direction.WEST
