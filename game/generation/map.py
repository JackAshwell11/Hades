"""Manages the procedural generation of the dungeon and places the player, enemies and
items into the game map."""
from __future__ import annotations

# Builtin
import logging
from collections import deque
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

# Pip
import numpy as np

# Custom
from game.constants.constructor import ENEMIES
from game.constants.general import DEBUG_LINES
from game.constants.generation import (
    BASE_ENEMY_COUNT,
    BASE_ITEM_COUNT,
    BASE_MAP_HEIGHT,
    BASE_MAP_WIDTH,
    BASE_ROOM,
    BASE_SPLIT_COUNT,
    ENEMY_DISTRIBUTION,
    ITEM_DISTRIBUTION,
    LARGE_ROOM,
    MAX_ENEMY_COUNT,
    MAX_ITEM_COUNT,
    MAX_MAP_HEIGHT,
    MAX_MAP_WIDTH,
    MAX_SPLIT_COUNT,
    PLACE_TRIES,
    SAFE_SPAWN_RADIUS,
    SMALL_ROOM,
    TileType,
)
from game.generation.bsp import Leaf, Point

if TYPE_CHECKING:
    from game.generation.bsp import Rect

__all__ = (
    "GameMapShape",
    "Map",
    "create_map",
)

# Get the logger
logger = logging.getLogger(__name__)

# Set the numpy print formatting to allow pretty printing (for debugging)
np.set_printoptions(threshold=10, edgeitems=30, linewidth=1000)


def create_map(level: int) -> tuple[np.ndarray, GameMapShape]:
    """Initialises and generates the game map.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Returns
    -------
    tuple[np.ndarray, GameMapShape]
        The generated map and a named tuple containing the width and height.
    """
    grid: np.ndarray = Map(level).grid
    return grid, GameMapShape(grid.shape[1], grid.shape[0])


class GameMapShape(NamedTuple):
    """Represents a two element tuple holding the width and height of a game map.

    Parameters
    ----------
    width: int
        The width of the game map.
    height: int
        The height of the game map.
    """

    width: int
    height: int


class Map:
    """Procedurally generates a game generation based on a given game level.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Attributes
    ----------
    map_constants: dict[TileType | str, int]
        A mapping of constant name to value. These constants are width, height, split
        count (how many times the bsp should split) and the counts for the different
        enemies and items.
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
    player_pos: tuple[int, int]
        The player's position in the grid. This is set to -1 to avoid typing errors.
    """

    __slots__ = (
        "level",
        "map_constants",
        "grid",
        "bsp",
        "player_spawn",
        "enemy_spawns",
        "probabilities",
        "player_pos",
    )

    def __init__(self, level: int) -> None:
        self.level: int = level
        self.map_constants: dict[TileType | str, int] = self._generate_constants()
        self.grid: np.ndarray | None = None
        self.bsp: Leaf | None = None
        self.player_spawn: tuple[int, int] | None = None
        self.enemy_spawns: list[tuple[int, int]] = []
        self.probabilities: dict[str, float] = BASE_ROOM
        self.player_pos: tuple[int, int] = (-1, -1)
        self.make_map()

    def __repr__(self) -> str:
        return (
            f"<Map (Width={self.map_constants['width']})"
            f" (Height={self.map_constants['height']}) (Split"
            f" count={self.map_constants['split count']}) (Enemy"
            f" count={self.map_constants['enemy count']}) (Item"
            f" count={self.map_constants['item count']})>"
        )

    @property
    def width(self) -> int:
        """Gets the width of the grid.

        Returns
        -------
        int
            The width of the grid.
        """
        # Make sure variables needed are valid
        assert self.grid is not None

        # Return the shape
        return self.grid.shape[1]

    @property
    def height(self) -> int:
        """Gets the height of the grid.

        Returns
        -------
        int
            The height of the grid.
        """
        # Make sure variables needed are valid
        assert self.grid is not None

        # Return the shape
        return self.grid.shape[0]

    def _generate_constants(self) -> dict[TileType | str, int]:
        """Generates the needed constants based on a given level.

        Returns
        -------
        dict[TileType | str, int]
            The generated constants.
        """
        # Create the generation constants
        generation_constants: dict[TileType | str, int] = {
            "width": np.minimum(
                int(np.round(BASE_MAP_WIDTH * 1.2**self.level)), MAX_MAP_WIDTH
            ),
            "height": np.minimum(
                int(np.round(BASE_MAP_HEIGHT * 1.2**self.level)), MAX_MAP_HEIGHT
            ),
            "split count": np.minimum(
                int(np.round(BASE_SPLIT_COUNT * 1.5**self.level)), MAX_SPLIT_COUNT
            ),
            "enemy count": np.minimum(
                int(np.round(BASE_ENEMY_COUNT * 1.1**self.level)), MAX_ENEMY_COUNT
            ),
            "item count": np.minimum(
                int(np.round(BASE_ITEM_COUNT * 1.1**self.level)), MAX_ITEM_COUNT
            ),
        }

        # Create the dictionary which will hold the counts for each enemy and item type
        type_dict: dict[TileType, int] = {
            key: int(np.ceil(value * generation_constants["enemy count"]))
            for key, value in ENEMY_DISTRIBUTION.items()
        } | {
            key: int(np.ceil(value * generation_constants["item count"]))
            for key, value in ITEM_DISTRIBUTION.items()
        }

        # Merge the enemy/item type count dict and the generation constants dict
        # together and then return the result
        result = generation_constants | type_dict
        logger.info("Generated map constants %r", result)
        return result

    def make_map(self) -> None:
        """Generates the map for a specified instance with a given level."""
        # Create the 2D grid used for representing the dungeon
        self.grid = np.full(
            (self.map_constants["height"], self.map_constants["width"]), 0, np.int8
        )

        # Create the first leaf used for generation
        self.bsp = Leaf(
            Point(0, 0),
            Point(self.map_constants["width"] - 1, self.map_constants["height"] - 1),
            None,
            self.grid,
        )

        # Start the splitting using a stack
        stack = deque["Leaf"]()
        stack.append(self.bsp)
        split_count = self.map_constants["split count"]
        while split_count:
            # Test if the stack is empty
            if not stack:
                break

            # Use the probabilities to check if we should split
            if np.random.choice([True, False], p=list(self.probabilities.values())):
                # Get the current leaf from the stack
                current = stack.pop()

                # Split the bsp
                if current.split(DEBUG_LINES) and current.left and current.right:
                    # Add the child leafs so they can be split
                    stack.append(current.left)
                    stack.append(current.right)

                # Multiply the probabilities by SMALL_ROOM
                self.probabilities["SMALL"] *= SMALL_ROOM["SMALL"]
                self.probabilities["LARGE"] *= SMALL_ROOM["LARGE"]
                logger.debug("Split bsp. New probabilities are %r", self.probabilities)

                # Decrement the split count
                split_count -= 1
            else:
                # Multiply the probabilities by LARGE_ROOM
                self.probabilities["SMALL"] *= LARGE_ROOM["SMALL"]
                self.probabilities["LARGE"] *= LARGE_ROOM["LARGE"]
                logger.debug(
                    "Didn't split bsp. New probabilities are %r", self.probabilities
                )

            # Normalise the probabilities, so they add up to 1
            probabilities_sum = 1 / sum(self.probabilities.values())  # type: ignore
            self.probabilities = {
                key: value * probabilities_sum
                for key, value in self.probabilities.items()
            }

        # Create the rooms. We can use the same stack since it is currently empty
        leafs: list[Leaf] = []
        stack.append(self.bsp)
        while stack:
            # Get the current leaf from the stack
            current = stack.pop()

            # Create the room
            result = current.create_room()
            if result:
                # Room creation successful so save the rect
                leafs.append(current)
            elif current.left and current.right:
                # Room creation not successful meaning there are child leafs so try
                # again on the child leafs
                stack.append(current.left)
                stack.append(current.right)

        # Get all the rooms objects from the leafs list, so we can store the hallways
        # too. To make the hallways, we can connect each pair of leaves in the leafs
        # list using itertools.pairwise
        rooms: list[Rect] = [leaf.room for leaf in leafs if leaf.room]
        hallways: list[Rect] = []
        logger.info("Created %d rooms", len(rooms))
        for pair in list(pairwise(leafs)):
            first_hallway, second_hallway = pair[0].create_hallway(pair[1])
            if first_hallway:
                hallways.append(first_hallway)
            if second_hallway:
                hallways.append(second_hallway)
        logger.info("Created %d hallways", len(hallways))

        # Create a sorted list of tuples based on the rect areas
        rects: list[Rect] = rooms + hallways
        rect_areas = sorted(
            ((rect, rect.width * rect.height) for rect in rects),
            key=lambda x: x[1],
        )
        total_area = sum(area[1] for area in rect_areas)
        logger.debug("Created %d total rects with area %d", len(rects), total_area)

        # Place the player spawn in the smallest room
        self._place_tile(TileType.PLAYER, rect_areas[0][0])

        # Place the enemies
        self._place_enemies(
            rect_areas,
            [area[1] / total_area for area in rect_areas],
        )

        # Place the items
        self._place_items(rect_areas)
        logger.info(
            "Finished creating game map with constants %r and rect count %d",
            self.map_constants,
            len(rects),
        )

    def _place_enemies(
        self,
        rect_areas: list[tuple[Rect, int]],
        area_probabilities: list[float],
    ) -> None:
        """Places the enemies in the grid making sure other tiles aren't replaced.

        Parameters
        ----------
        rect_areas: list[tuple[Rect, int]]
            A sorted list of rects and their areas.
        area_probabilities: list[float]
            A list of areas probabilities. This corresponds to rect_areas.
        """
        # Repeatedly place an enemy type. If they are placed, we can increment the
        # counter. Otherwise, continue
        for enemy in ENEMY_DISTRIBUTION:
            # Set up the counters for this enemy type
            count = self.map_constants[enemy]
            enemies_placed = 0
            tries = PLACE_TRIES
            while enemies_placed < count and tries != 0:
                if self._place_tile(
                    enemy,
                    np.random.choice(
                        [rect[0] for rect in rect_areas], p=area_probabilities
                    ),
                ):
                    # Enemy placed
                    enemies_placed += 1
                else:
                    # Enemy not placed
                    tries -= 1

    def _place_items(self, rect_areas: list[tuple[Rect, int]]) -> None:
        """Places the items in the grid making sure other tiles aren't replaced.

        Parameters
        ----------
        rect_areas: list[tuple[Rect, int]]
            A sorted list of rects and their areas. This is only used to pick a random
            rect, the items aren't actually placed based on weights.
        """
        # Repeatedly place an item type. If they are placed, we can increment the
        # counter. Otherwise, continue
        for item in ITEM_DISTRIBUTION:
            # Set up the counters for this item type
            count = self.map_constants[item]
            items_placed = 0
            tries = PLACE_TRIES
            while items_placed < count and tries != 0:
                if self._place_tile(
                    item, np.random.choice([rect[0] for rect in rect_areas])
                ):
                    # Item placed
                    items_placed += 1
                else:
                    # Item not placed
                    tries -= 1

    def _place_tile(self, entity: TileType, rect: Rect) -> bool:
        """Places a given entity in a random position in a given rect.

        Parameters
        ----------
        entity: TileType
            The entity to place in the grid.
        rect: Rect
            The rect object to place the tile in.

        Returns
        -------
        bool
            Whether or not an enemy was placed.
        """
        # Make sure variables needed are valid
        assert self.grid is not None

        # Get a random position within the rect making sure to exclude the walls
        position_x, position_y = (
            np.random.randint(rect.top_left.x + 1, rect.bottom_right.x - 1),
            np.random.randint(rect.top_left.y + 1, rect.bottom_right.y - 1),
        )

        # Check if the entity is an enemy. If so, we need to make sure they are not
        # within the spawn radius
        if entity in ENEMIES:
            distance_to_player = np.hypot(
                self.player_pos[0] - position_x, self.player_pos[1] - position_y
            )
            if distance_to_player < SAFE_SPAWN_RADIUS:
                # Enemy is within spawn radius so don't place them
                return False

        # Check if the chosen position is already taken
        if self.grid[position_y, position_x] != TileType.FLOOR.value:
            # Already taken
            return False

        # Place the entity in the random position
        self.grid[position_y, position_x] = entity.value

        # Check if the entity is the player. If so, save the position
        if entity is TileType.PLAYER:
            self.player_pos = (position_x, position_y)

        # Return true so we know an enemy has been placed
        return True
