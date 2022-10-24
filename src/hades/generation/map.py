"""Manages the generation of the dungeon and placing of game objects."""
from __future__ import annotations

# Builtin
import logging
from collections import deque
from heapq import heappop, heappush
from itertools import permutations
from typing import TYPE_CHECKING, NamedTuple

# Pip
import numpy as np

# Custom
from hades.constants.generation import (
    EXTRA_MAXIMUM_PERCENTAGE,
    HALLWAY_SIZE,
    ITEM_DISTRIBUTION,
    ITEM_PLACE_TRIES,
    MAP_GENERATION_COUNTS,
    REMOVED_CONNECTION_LIMIT,
    GenerationConstantType,
    TileType,
)
from hades.extensions import calculate_astar_path
from hades.generation.bsp import Leaf
from hades.generation.primitives import Point, Rect

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = (
    "LevelConstants",
    "Map",
    "create_map",
)

# Get the logger
logger = logging.getLogger(__name__)

# Set the numpy print formatting to allow pretty printing (for debugging)
np.set_printoptions(threshold=1, edgeitems=50, linewidth=10000)


def create_map(level: int) -> tuple[Map, LevelConstants]:
    """Generate the game map for a given game level.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Returns
    -------
    tuple[Map, LevelConstants]
        The generated map and a named tuple containing the width and height.
    """
    # Create the rooms
    temp_map = Map(level)
    rooms = temp_map.split_bsp().generate_rooms()

    # Create the connections for the hallways
    complete_graph: dict[Rect, list[Rect]] = {}
    for source, destination in permutations(rooms, 2):
        complete_graph.update({source: complete_graph.get(source, []) + [destination]})
    temp_map.create_hallways(
        temp_map.add_extra_connections(
            complete_graph, temp_map.create_mst(complete_graph)
        )
    )

    # f = {}
    # i = 0
    # for key, value in complete_graph.items():
    #     if f.get(key, -1) == -1:
    #         f[key] = i
    #         i += 1
    #     for j in value:
    #         if f.get(j, -1) == -1:
    #             f[j] = i
    #             i += 1
    #
    # print([f"{room.center} - {f[room]}" for room in rooms])
    # print([(f[i[1]], f[i[2]]) for i in mst])
    # print([(f[i[1]], f[i[2]]) for i in hallway_connections])
    # for i in [[i.value for i in row] for row in self.grid]:
    #     print(i)

    # Place the game objects
    possible_tiles: list[tuple[int, int]] = list(  # noqa
        zip(*np.nonzero(temp_map.grid == TileType.FLOOR))
    )
    np.random.shuffle(possible_tiles)
    temp_map.place_tile(TileType.PLAYER, possible_tiles)
    temp_map.place_multiple(ITEM_DISTRIBUTION, possible_tiles)

    # Return the map object and a GameMapShape object
    return temp_map, LevelConstants(
        level, temp_map.grid.shape[1], temp_map.grid.shape[0]
    )


class LevelConstants(NamedTuple):
    """Holds the constants for a specific level.

    level: int
        The level of this game.
    width: int
        The width of the game map.
    height: int
        The height of the game map.
    """

    level: int
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
    map_constants: dict[TileType | GenerationConstantType, int]
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
        self.map_constants: dict[
            TileType | GenerationConstantType, int
        ] = self.generate_constants()
        self.grid: np.ndarray = np.full(
            (
                self.map_constants[GenerationConstantType.HEIGHT],
                self.map_constants[GenerationConstantType.WIDTH],
            ),
            TileType.EMPTY,  # type: ignore
            TileType,
        )
        self.bsp: Leaf = Leaf(
            Point(0, 0),
            Point(
                self.map_constants[GenerationConstantType.WIDTH] - 1,
                self.map_constants[GenerationConstantType.HEIGHT] - 1,
            ),
            self.grid,
        )
        self.player_pos: tuple[int, int] = (-1, -1)

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return (
            f"<Map (Width={self.map_constants[GenerationConstantType.WIDTH]})"
            f" (Height={self.map_constants[GenerationConstantType.HEIGHT]}) (Split"
            f" count={self.map_constants[GenerationConstantType.SPLIT_ITERATION]})"
            f" (Item count={self.map_constants[GenerationConstantType.ITEM_COUNT]})>"
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

    def generate_constants(self) -> dict[TileType | GenerationConstantType, int]:
        """Generate the needed constants based on a given game level.

        Returns
        -------
        dict[TileType | GenerationConstantType, int]
            The generated constants.
        """
        # Create the generation constants
        generation_constants: dict[TileType | GenerationConstantType, int] = {
            key: np.minimum(
                int(np.round(value.base_value * value.increase**self.level)),
                value.max_value,
            )
            for key, value in MAP_GENERATION_COUNTS.items()
        }

        # Create the dictionary which will hold the counts for each item type
        item_dict: dict[TileType, int] = {
            key: int(
                np.round(
                    value * generation_constants[GenerationConstantType.ITEM_COUNT]
                )
            )
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
        split_iteration = self.map_constants[GenerationConstantType.SPLIT_ITERATION]
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

    def generate_rooms(self) -> set[Rect]:
        """Generate the rooms for a given game level using the bsp.

        Returns
        -------
        set[Rect]
            The generated rooms.
        """
        # Create the rooms. We can use the same deque object since it is currently empty
        rooms: set[Rect] = set()
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
                rooms.add(current.room)

        # Return all the created rooms
        return rooms

    @staticmethod
    def create_mst(
        complete_graph: dict[Rect, list[Rect]]
    ) -> set[tuple[float, Rect, Rect]]:
        """Create a minimum spanning tree from a set of rects using Prim's algorithm.

        Further reading which may be useful:
        `Prim's algorithm <https://en.wikipedia.org/wiki/Prim's_algorithm>`_

        Parameters
        ----------
        complete_graph: dict[Rect, list[Rect]]
            An adjacency list which represents a complete graph.

        Returns
        -------
        set[tuple[float, Rect, Rect]]
            A set of connections and their costs which form the minimum spanning tree.
        """
        # Use Prim's algorithm to construct a minimum spanning tree from complete_graph
        start = next(iter(complete_graph))
        visited: set[Rect] = set()
        unexplored: list[tuple[float, Rect, Rect]] = [(0, start, start)]
        mst: set[tuple[float, Rect, Rect]] = set()
        while len(mst) < len(complete_graph) - 1:
            # Get the neighbour with the lowest cost
            cost, source, destination = heappop(unexplored)

            # Check if the neighbour is already visited or not
            if destination in visited:
                continue

            # Neighbour isn't visited so mark them as visited and add their neighbours
            # to the heap
            visited.add(destination)
            for neighbour in complete_graph[destination]:
                if neighbour not in visited:
                    heappush(
                        unexplored,
                        (
                            destination.get_distance_to(neighbour),
                            destination,
                            neighbour,
                        ),
                    )

            # Add a new edge towards the lowest cost neighbour onto the mst
            if source != destination:
                mst.add((cost, source, destination))

        # Return the mst
        return mst

    @staticmethod
    def add_extra_connections(
        complete_graph: dict[Rect, list[Rect]], mst: set[tuple[float, Rect, Rect]]
    ) -> set[tuple[float, Rect, Rect]]:
        """Add extra connections back into the minimum spanning tree.

        Parameters
        ----------
        complete_graph: dict[Rect, list[Rect]]
            An adjacency list which represents a complete graph.
        mst: set[tuple[float, Rect, Rect]]
            The minimum spanning tree to add connections too.

        Returns
        -------
        set[tuple[float, Rect, Rect]]
            The expanded minimum spanning tree with more connections.
        """
        # Find the maximum cost that an extra connection can be. This is the maximum
        # cost a mst connection is + EXTRA_CONNECTION_PERCENTAGE
        maximum_mst_cost = (
            sorted(mst, key=lambda mst_cost: mst_cost[0], reverse=True)[0][0]
            * EXTRA_MAXIMUM_PERCENTAGE
        )

        # Delete every connection that will create a symmetric relation and whose cost
        # is greater than the maximum_mst_cost
        mst_non_symmetric: set[tuple[float, Rect, Rect]] = set()
        for source, connections in complete_graph.items():
            for connection in connections:
                # Get the distance between the two rooms and check if its greater or
                # equal to maximum_mst_cost
                connection_cost = source.get_distance_to(connection)
                if connection_cost >= maximum_mst_cost:
                    continue

                # Check if this connection will create a symmetric relation. If not,
                # save it
                complete_relation = connection_cost, source, connection
                if complete_relation not in mst and (
                    connection_cost,
                    connection,
                    source,
                ) not in mst.union(mst_non_symmetric):
                    mst_non_symmetric.add(complete_relation)

        # Add some removed connections back into the graph and return the result, so the
        # dungeon is not as sparsely populated
        return mst.union(
            {
                mst_non_symmetric.pop()
                for _ in range(round(len(mst_non_symmetric) * REMOVED_CONNECTION_LIMIT))
            }
        )

    def create_hallways(self, connections: set[tuple[float, Rect, Rect]]) -> Map:
        """Create the hallways by placing random obstacles and pathfinding around them.

        Parameters
        ----------
        connections: set[Rects]
            The connections to pathfind using the A* algorithm.

        Returns
        -------
        Map
            The map object which represents the dungeon.
        """
        # Place random obstacles in the grid
        y, x = np.where(self.grid == TileType.EMPTY)
        arr_index = np.random.choice(
            len(y), self.map_constants[GenerationConstantType.OBSTACLE_COUNT]
        )
        self.grid[y[arr_index], x[arr_index]] = TileType.OBSTACLE
        logger.debug(
            "Created %d obstacles in the 2D grid",
            self.map_constants[GenerationConstantType.OBSTACLE_COUNT],
        )

        # Use the A* algorithm with to connect each pair of rooms making sure to avoid
        # the obstacles giving us natural looking hallways. Note that the width of the
        # hallways will always be odd in this implementation due to numpy indexing
        half_hallway_size = HALLWAY_SIZE // 2
        for _, pair_source, pair_destination in connections:
            for path_point_tup in calculate_astar_path(
                self.grid,
                pair_source.center,
                pair_destination.center,
            ):
                # Test if the current tile is a floor tile
                path_point = Point(*path_point_tup)
                if self.grid[path_point.y][path_point.x] is TileType.FLOOR:
                    # Current tile is a floor tile, so there is no point placing a rect
                    continue

                # Place a rect box around the path_point using HALLWAY_SIZE to determine
                # the width and height
                logger.debug(
                    "Creating path from %r to %r", pair_source, pair_destination
                )
                Rect(
                    Point(
                        path_point.x - half_hallway_size,
                        path_point.y - half_hallway_size,
                    ),
                    Point(
                        path_point.x + half_hallway_size,
                        path_point.y + half_hallway_size,
                    ),
                ).place_rect(self.grid)

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
