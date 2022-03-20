from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from generation.bsp import Rect


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
            if np.random.choice([True, False], p=list(self.probabilities.values())):
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
            probabilities_sum = 1 / sum(self.probabilities.values())  # type: ignore
            self.probabilities = {
                key: value * probabilities_sum
                for key, value in self.probabilities.items()
            }

        # Create a list which will hold all the created rects. Since mutable data
        # structures in Python don't create new objects when passed as parameters,
        # modifying the list in each recursive call will modify this original list
        # simplifying the code
        rects: list[Rect] = []

        # Create the rooms and hallways recursively
        self.bsp.create_room(rects)
        self.bsp.create_hallway(rects)

        # Create a sorted list of tuples based on the rect areas
        rect_areas = sorted(
            ((rect, rect.width * rect.height) for rect in rects),
            key=lambda x: x[1],
        )

        # Place the player spawn in the smallest room
        self.place_tile(TileType.PLAYER, rect_areas[0][0])

        # Get the total area
        areas = [area[1] for area in rect_areas]
        total_area = sum(areas)

        # Get all the destination rects
        for rect in np.random.choice(
            [rect[0] for rect in rect_areas],
            self.map_constants["enemy count"],
            p=[area / total_area for area in areas],
        ):
            self.place_tile(TileType.ENEMY, rect)

    def place_tile(self, entity: TileType, rect: Rect) -> None:
        """
        Places a given entity in a random position in a given rect.

        Parameters
        ----------
        entity: TileType
            The entity to place in the grid.
        rect: Rect
            The rect object to place the tile in.
        """
        # Make sure variables needed are valid
        assert self.grid is not None

        # Get a random position within the rect making sure to exclude the walls
        position_x, position_y = (
            np.random.randint(rect.x1 + 1, rect.x2 - 1),
            np.random.randint(rect.y1 + 1, rect.y2 - 1),
        )

        # Place the entity in the random position
        self.grid[position_y, position_x] = entity.value
