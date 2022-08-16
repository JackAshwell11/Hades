"""Manages the generation of the dungeon and placing of game objects."""
from __future__ import annotations

# Builtin
import logging
from collections import deque
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

# Pip
import numpy as np

# Custom
from hades.constants.generation import (
    BASE_ITEM_COUNT,
    BASE_MAP_HEIGHT,
    BASE_MAP_WIDTH,
    BASE_OBSTACLE_COUNT,
    BASE_SPLIT_ITERATION,
    HALLWAY_SIZE,
    ITEM_DISTRIBUTION,
    ITEM_PLACE_TRIES,
    MAX_ITEM_COUNT,
    MAX_MAP_HEIGHT,
    MAX_MAP_WIDTH,
    MAX_OBSTACLE_COUNT,
    MAX_SPLIT_ITERATION,
    TileType,
)
from hades.generation.astar import calculate_astar_path
from hades.generation.bsp import Leaf
from hades.generation.primitives import Point, Rect

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = (
    "GameMapShape",
    "Map",
    "create_map",
)

# Get the logger
logger = logging.getLogger(__name__)

# Set the numpy print formatting to allow pretty printing (for debugging)
np.set_printoptions(threshold=1, edgeitems=50, linewidth=10000)


def create_map(level: int) -> tuple[Map, GameMapShape]:
    """Generate the game map for a given game level.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Returns
    -------
    tuple[Map, GameMapShape]
        The generated map and a named tuple containing the width and height.
    """
    # Create the rooms and hallways
    temp_map = Map(level)
    temp_map.create_hallways(temp_map.split_bsp().generate_rooms())

    # Place the game objects
    possible_tiles: list[tuple[int, int]] = list(  # noqa
        zip(*np.nonzero(temp_map.grid == TileType.FLOOR))
    )
    np.random.shuffle(possible_tiles)
    temp_map.place_tile(TileType.PLAYER, possible_tiles)
    temp_map.place_multiple(ITEM_DISTRIBUTION, possible_tiles)

    # Return the map object and a GameMapShape object
    return temp_map, GameMapShape(temp_map.grid.shape[1], temp_map.grid.shape[0])


class GameMapShape(NamedTuple):
    """Represents a two element tuple holding the width and height of a game map.

    width: int
        The width of the game map.
    height: int
        The height of the game map.
    """

    width: int
    height: int


class Map:
    """Procedurally generates a dungeon map based on a given game level.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Attributes
    ----------
    map_constants: dict[TileType | str, int]
        A mapping of constant name to value. These constants are 'width', 'height',
        'split iteration' (how many times the bsp should split) and the counts for each
        item type (accessed through TileType).
    grid: np.ndarray
        The 2D grid which represents the dungeon.
    bsp: Leaf
        The root leaf for the binary space partition.
    player_pos: tuple[int, int]
        The player's position in the grid. This is set to (-1, -1) to avoid typing
        errors.
    """

    __slots__ = (
        "level",
        "map_constants",
        "grid",
        "bsp",
        "player_pos",
    )

    def __init__(self, level: int) -> None:
        self.level: int = level
        self.map_constants: dict[TileType | str, int] = self.generate_constants()
        self.grid: np.ndarray = np.full(
            (self.map_constants["height"], self.map_constants["width"]),
            TileType.EMPTY,
            TileType,
        )
        self.bsp: Leaf = Leaf(
            Point(0, 0),
            Point(self.map_constants["width"] - 1, self.map_constants["height"] - 1),
            self.grid,
        )
        self.player_pos: tuple[int, int] = (-1, -1)

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return (
            f"<Map (Width={self.map_constants['width']})"
            f" (Height={self.map_constants['height']}) (Split"
            f" count={self.map_constants['split iteration']}) (Item"
            f" count={self.map_constants['item count']})>"
        )

    @property
    def width(self) -> int:
        """Get the width of the grid.

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
        """Get the height of the grid.

        Returns
        -------
        int
            The height of the grid.
        """
        # Make sure variables needed are valid
        assert self.grid is not None

        # Return the shape
        return self.grid.shape[0]

    def generate_constants(self) -> dict[TileType | str, int]:
        """Generate the needed constants based on a given game level.

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
            "split iteration": np.minimum(
                int(np.round(BASE_SPLIT_ITERATION * 1.5**self.level)),
                MAX_SPLIT_ITERATION,
            ),
            "obstacle count": np.minimum(
                int(np.round(BASE_OBSTACLE_COUNT * 1.3**self.level)),
                MAX_OBSTACLE_COUNT,
            ),
            "item count": np.minimum(
                int(np.round(BASE_ITEM_COUNT * 1.1**self.level)), MAX_ITEM_COUNT
            ),
        }

        # Create the dictionary which will hold the counts for each item type
        item_dict: dict[TileType, int] = {
            key: int(np.round(value * generation_constants["item count"]))
            for key, value in ITEM_DISTRIBUTION.items()
        }

        # Merge the generation constants dict and the item type count dict together and
        # then return the result
        result = generation_constants | item_dict
        logger.info("Generated map constants %r", result)
        return result

    def split_bsp(self) -> Map:
        """Split the bsp based on the generated constants.

        Returns
        -------
        Map
            The map object which represents the dungeon.
        """
        # Start the splitting using deque
        deque_obj = deque["Leaf"]()
        deque_obj.append(self.bsp)
        split_iteration = self.map_constants["split iteration"]
        while split_iteration and deque_obj:
            # Get the current leaf from the deque object
            current = deque_obj.popleft()

            # Split the bsp if possible
            if current.split() and current.left and current.right:
                # Add the child leafs so they can be split
                logger.debug("Split bsp. Split iteration is now %d", split_iteration)
                deque_obj.append(current.left)
                deque_obj.append(current.right)

                # Decrement the split iteration
                split_iteration -= 1

        # Return the map object
        return self

    def generate_rooms(self) -> list[Rect]:
        """Generate the rooms for a given game level using the bsp.

        Returns
        -------
        list[Rect]
            The generated rooms.
        """
        # Create the rooms. We can use the same deque object since it is currently empty
        rooms: list[Rect] = []
        deque_obj = deque["Leaf"]()
        deque_obj.append(self.bsp)
        while deque_obj:
            # Get the current leaf from the stack
            current = deque_obj.pop()

            # Check if a room already exists in this leaf
            if current.room:
                continue

            # Test if we can create a room in the current leaf
            if current.left and current.right:
                # Room creation not successful meaning there are child leafs so try
                # again on the child leafs
                deque_obj.append(current.left)
                deque_obj.append(current.right)
            else:
                # Create a room in the current leaf and save the rect
                logger.debug("Creating room in leaf %r", current)
                while not current.create_room():
                    # Width to height ratio is outside of range so try again
                    logger.debug("Trying generation of room in leaf %r again", current)

                # Add the created room to the rooms list
                assert current.room  # Have to make sure its actually created first
                rooms.append(current.room)

        # Return all the created rooms
        return rooms

    def create_hallways(self, rooms: list[Rect]) -> Map:
        """Create the hallways by placing random obstacles and pathfinding around them.

        Parameters
        ----------
        rooms: list[Rects]
            The rooms to make hallways between using the A* algorithm.

        Returns
        -------
        Map
            The map object which represents the dungeon.
        """
        # Place random obstacles in the grid
        y, x = np.where(self.grid == TileType.EMPTY)
        arr_index = np.random.choice(len(y), self.map_constants["obstacle count"])
        self.grid[y[arr_index], x[arr_index]] = TileType.OBSTACLE
        logger.debug(
            "Created %d obstacles in the 2D grid", self.map_constants["obstacle count"]
        )

        # # Create a complete graph out of rooms
        # connections: set[tuple[Point, Point, float]] = set()
        # complete_graph: dict[Point, list[tuple[Point, float]]] = {}
        # from itertools import permutations
        # from heapq import heappop, heappush
        # for source, destination in permutations(rooms, 2):
        #     cost = source.get_distance_to(destination)
        #     source_center = source.center
        #     destination_center = destination.center
        #     complete_graph.update(
        #         {
        #             source_center: complete_graph.get(source_center, [])
        #             + [(destination_center, cost)]
        #         }
        #     )
        #     connections.add((source_center, destination_center, cost))
        #
        # # Use Prim's algorithm to construct a minimum spanning tree of complete_graph
        # visited: set[Point] = set()
        # start = next(iter(complete_graph))
        # unexplored: list[tuple[float, Point, Point]] = [(0, start, start)]
        # mst: set[tuple[Point, Point, float]] = set()
        # while unexplored:
        #     # Get the neighbour with the lowest cost
        #     cost, source, destination = heappop(unexplored)
        #
        #     # Check if the neighbour is already visited or not
        #     if destination not in visited:
        #         # Neighbour isn't visited so mark them as visited and add their
        #         # neighbours to the heap
        #         visited.add(destination)
        #         for neighbour, neighbour_cost in complete_graph[destination]:
        #             if neighbour not in visited:
        #                 heappush(unexplored, (neighbour_cost, destination, neighbour))
        #
        #         # Add a new edge towards the lowest cost neighbour onto the mst
        #         if source != destination:
        #             mst.add((source, destination, cost))
        #
        # # Add some removed edges back into the graph, so it's not as sparsely
        # # populated
        # removed_edges = connections - connections.intersection(mst)
        # hallway_connections = mst.copy().union(
        #     {
        #         removed_edges.pop()
        #         for _ in range(round(len(removed_edges) * 0.15))
        #     }
        # )

        # Use the A* algorithm with to connect each pair of rooms making sure to avoid
        # the obstacles giving us natural looking hallways. Note that the width of the
        # hallways will always be odd in this implementation due to numpy indexing
        half_hallway_size = HALLWAY_SIZE // 2
        for pair_source, pair_destination in pairwise(rooms):
            for path_point in calculate_astar_path(
                self.grid,
                Point(*pair_source.center),
                Point(*pair_destination.center),
            ):
                # Test if the current tile is a floor tile
                if self.grid[path_point.y][path_point.x] is TileType.FLOOR:
                    # Current tile is a floor tile, so there is no point placing a rect
                    continue

                # Place a rect box around the path_point using HALLWAY_SIZE to determine
                # the width and height
                logger.debug(
                    "Creating path from %r to %r", pair_source, pair_destination
                )
                Rect(
                    self.grid,
                    Point(
                        path_point.x - half_hallway_size,
                        path_point.y - half_hallway_size,
                    ),
                    Point(
                        path_point.x + half_hallway_size,
                        path_point.y + half_hallway_size,
                    ),
                ).place_rect()

        # Return the map object
        return self

    def place_multiple(
        self,
        target_distribution: Mapping[TileType, int | float],
        possible_tiles: list[tuple[int, int]],
    ) -> Map:
        """Places multiple tile types from a given distribution in the 2D grid.

        Parameters
        ----------
        target_distribution: Mapping[TileType, int | float]
            The target distribution to place in the 2D grid.
        possible_tiles: list[tuple[int, int]]
            The possible tiles that the tiles can be placed into.

        Returns
        -------
        Map
            The map object which represents the dungeon.
        """
        # Place each tile in the distribution based on their probabilities of occurring
        for tile in target_distribution:
            tiles_placed = 0
            tries = ITEM_PLACE_TRIES
            while tiles_placed < self.map_constants[tile] and tries != 0:
                if self.place_tile(tile, possible_tiles):
                    # Tile placed
                    logger.debug("One of multiple %r placed in the 2D grid", tile)
                    tiles_placed += 1
                else:
                    # Tile not placed
                    logger.debug("Can't place one of multiple %r in the 2D grid", tile)
                    tries -= 1

        # Return the map object
        return self

    def place_tile(
        self, target_tile: TileType, possible_tiles: list[tuple[int, int]]
    ) -> bool:
        """Places a given tile in the 2D grid.

        Parameters
        ----------
        target_tile: TileType
            The tile to place in the 2D grid.
        possible_tiles: list[tuple[int, int]]
            The possible tiles that the tile can be placed into.

        Returns
        -------
        bool
            Whether the tile was placed or not.
        """
        # Check if there are any floor tiles left
        if possible_tiles:
            # Get a random floor position and place the target tile
            y, x = possible_tiles.pop()
            self.grid[y][x] = target_tile

            # Check if the target tile is the player. If so, we need to store its
            # position
            if target_tile is TileType.PLAYER:
                self.player_pos = x, y

            # Placing successful so return True
            logger.debug("Placed tile %r in the 2D grid")
            return True
        else:
            # Placing not successful so return False
            logger.debug("Can't place tile %r in the 2D grid")
            return False
