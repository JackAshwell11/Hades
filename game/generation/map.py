from __future__ import annotations

# Pip
import numpy as np
from constants import (
    BASE_ENEMY_COUNT,
    BASE_MAP_HEIGHT,
    BASE_MAP_WIDTH,
    BASE_ROOM,
    BASE_SPLIT_COUNT,
    DEBUG_LINES,
    LARGE_ROOM,
    SMALL_ROOM,
    TileType,
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
    map_constants: dict[str, int]
        A mapping of constant name to value. These constants are width, height, split
        count (how many times the bsp should split) and enemy count.
    grid: np.ndarray | None
        The 2D grid which represents the dungeon.
    bsp: Leaf | None
        The root leaf for the binary space partition.
    player_spawn: tuple[int, int] | None
        The coordinates for the player spawn. This is in the format (x, y).
    enemy_spawns: list[tuple[int, int]]
        The coordinates for the enemy spawn points. This is in the format (x, y).
    probabilities: dict[str, float]
        The current probabilities used to determine if we should split on each iteration
        or not.
    """

    def __init__(self, level: int) -> None:
        self.level: int = level
        self.map_constants: dict[str, int] = self.generate_constants()
        self.grid: np.ndarray | None = None
        self.bsp: Leaf | None = None
        self.player_spawn: tuple[int, int] | None = None
        self.enemy_spawns: list[tuple[int, int]] = []
        self.probabilities: dict[str, float] = BASE_ROOM
        self.make_map()

    def __repr__(self) -> str:
        return (
            f"<Map (Width={self.map_constants['width']})"
            f" (Height={self.map_constants['height']}) (Split"
            f" count={self.map_constants['split count']}) (Enemy"
            f" count={self.map_constants['enemy count']})>"
        )

    def generate_constants(self) -> dict[str, int]:
        """
        Generates the needed constants based on a given level.

        Returns
        -------
        dict[str, int]
            The generated constants.
        """
        return {
            "width": int(np.ceil(BASE_MAP_WIDTH * 1.2**self.level)),
            "height": int(np.ceil(BASE_MAP_HEIGHT * 1.2**self.level)),
            "split count": int(np.ceil(BASE_SPLIT_COUNT * 1.5**self.level)),
            "enemy count": int(np.ceil(BASE_ENEMY_COUNT * 1.1**self.level)),
        }

    def make_map(self) -> None:
        """Function which manages the map generation for a specified level."""
        # Set the numpy print formatting to allow pretty printing (for debugging)
        np.set_printoptions(
            edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.3g" % x)
        )

        # Create the 2D grid used for representing the dungeon
        self.grid = np.full(
            (self.map_constants["height"], self.map_constants["width"]), 0, np.int8
        )

        # Create the first leaf used for generation
        self.bsp = Leaf(
            0,
            0,
            self.map_constants["width"] - 1,
            self.map_constants["height"] - 1,
            self.grid,
        )

        # Start the recursive splitting
        for count in range(self.map_constants["split count"]):
            # Use the probabilities to check if we should split
            if np.random.choice(
                [True, False],
                p=[self.probabilities["SMALL"], self.probabilities["LARGE"]],
            ):
                # Split the bsp
                self.bsp.split(DEBUG_LINES)

                # Multiply the probabilities by SMALL_ROOM
                self.probabilities["SMALL"] *= SMALL_ROOM["SMALL"]
                self.probabilities["LARGE"] *= SMALL_ROOM["LARGE"]
            else:
                # Multiply the probabilities by LARGE_ROOM
                self.probabilities["SMALL"] *= LARGE_ROOM["SMALL"]
                self.probabilities["LARGE"] *= LARGE_ROOM["LARGE"]

            # Normalise the probabilities so they add up to 1
            probabilities_sum = 1 / (
                self.probabilities["SMALL"] + self.probabilities["LARGE"]
            )
            self.probabilities["SMALL"] = (
                self.probabilities["SMALL"] * probabilities_sum
            )
            self.probabilities["LARGE"] = (
                self.probabilities["LARGE"] * probabilities_sum
            )

        # Create the rooms recursively
        self.bsp.create_room()

        # Create the hallways recursively
        self.bsp.create_hallway()

        # Get the coordinates for the player spawn
        self.replace_random_floor(TileType.PLAYER)

        # Get the coordinates for the enemy spawn points
        for _ in range(self.map_constants["enemy count"]):
            self.replace_random_floor(TileType.ENEMY)

    def replace_random_floor(self, entity: TileType) -> None:
        """
        Replaces a random floor tile with a player, enemy or item tile.

        Parameters
        ----------
        entity: TileType
            The tile to replace the floor tile with.
        """
        assert self.grid is not None
        spaces = np.argwhere(self.grid == 1)
        player_spawn = spaces[np.random.randint(0, len(spaces))]
        self.grid[player_spawn[0], player_spawn[1]] = entity.value
