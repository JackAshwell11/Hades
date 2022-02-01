from __future__ import annotations

# Builtin
import random
from typing import List, Optional, Tuple

# Pip
import numpy as np
from constants import (
    DEBUG_LINES,
    ENEMY,
    ENEMY_COUNT,
    MAP_HEIGHT,
    MAP_WIDTH,
    PLAYER,
    SPLIT_COUNT,
)

# Custom
from generation.bsp import Leaf


class Map:
    """
    Procedurally generates a game generation based on a given game level.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Attributes
    ----------
    width: int
        The width of the game generation.
    height: int
        The height of the game generation.
    split_count: int = 5
        The amount of times the bsp should split.
    grid: Optional[np.ndarray]
        The 2D grid which represents the dungeon.
    bsp: Optional[Leaf]
        The root leaf for the binary space partition.
    player_spawn: Optional[Tuple[int, int]]
        The coordinates for the player spawn. This is in the format (x, y).
    enemy_spawns: List[Tuple[int, int]]
        The coordinates for the enemy spawn points. This is in the format (x, y).
    """

    def __init__(self, level: int) -> None:
        self.level: int = level
        self.width: int = MAP_WIDTH
        self.height: int = MAP_HEIGHT
        self.split_count: int = SPLIT_COUNT
        self.grid: Optional[np.ndarray] = None
        self.bsp: Optional[Leaf] = None
        self.player_spawn: Optional[Tuple[int, int]] = None
        self.enemy_spawns: List[Tuple[int, int]] = []
        self.make_map()

    def __repr__(self) -> str:
        return f"<Map (Width={self.width}) (Height={self.height})>"

    def make_map(self) -> None:
        """Function which manages the map generation for a specified level."""
        # Set the numpy print formatting to allow pretty printing (for debugging)
        np.set_printoptions(
            edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.3g" % x)
        )

        # Create the 2D grid used for representing the dungeon
        self.grid = np.full((self.height, self.width), 0, np.int8)

        # Create the first leaf used for generation
        self.bsp = Leaf(0, 0, self.width - 1, self.height - 1, self.grid)

        # Start the recursive splitting
        for count in range(self.split_count):
            self.bsp.split(DEBUG_LINES)

        # Create the rooms recursively
        self.bsp.create_room()

        # Create the hallways recursively
        self.bsp.create_hallway()

        # Get the coordinates for the player spawn
        self.replace_random_floor(PLAYER)

        # Get the coordinates for the enemy spawn points
        for _ in range(ENEMY_COUNT):
            self.replace_random_floor(ENEMY)

    def replace_random_floor(self, entity: int) -> None:
        """
        Replaces a random floor tile with a player, enemy or item tile.

        Parameters
        ----------
        entity: int
            The tile to replace the floor tile with.
        """
        assert self.grid is not None
        player_spawn = random.choice(np.argwhere(self.grid == 1))
        self.grid[player_spawn[0], player_spawn[1]] = entity
