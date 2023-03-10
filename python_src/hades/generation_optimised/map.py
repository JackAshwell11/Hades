"""Manages the generation of the dungeon and placing of game objects."""
from __future__ import annotations

# Builtin
import logging
import random
from collections import deque
from heapq import heappop, heappush
from itertools import permutations
from typing import NamedTuple

# Pip
import numpy as np
import numpy.typing as npt

# Custom
from hades.constants.generation import (
    HALLWAY_SIZE,
    ITEM_DISTRIBUTION,
    MAP_GENERATION_COUNTS,
    GenerationConstantType,
    TileType,
)
from hades.extensions import calculate_astar_path
from hades.generation_optimised.bsp import Leaf
from hades.generation_optimised.primitives import Point, Rect

__all__ = (
    "LevelConstants",
    "create_hallways",
    "create_map",
    "create_mst",
    "generate_constants",
    "generate_rooms",
    "place_tile",
    "split_bsp",
)

# Get the logger
logger = logging.getLogger(__name__)

# Set the numpy print formatting to allow pretty printing (for debugging)
np.set_printoptions(threshold=1, edgeitems=50, linewidth=10000)


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


def generate_constants(level: int) -> dict[TileType | GenerationConstantType, int]:
    """Generate the needed constants based on a given game level.

    Parameters
    ----------
    level: int
        The game level to generate constants for.

    Returns
    -------
    dict[TileType | GenerationConstantType, int]
        The generated constants.
    """
    # Create the generation constants
    generation_constants: dict[GenerationConstantType, int] = {
        key: min(
            int(round(value.base_value * value.increase**level)),
            value.max_value,
        )
        for key, value in MAP_GENERATION_COUNTS.items()
    }

    # Create the dictionary which will hold the counts for each item type
    item_dict: dict[TileType, int] = {
        key: int(round(value * generation_constants[GenerationConstantType.ITEM_COUNT]))
        for key, value in ITEM_DISTRIBUTION.items()
    }

    # Merge the generation constants dict and the item type count dict together and
    # then return the result
    result = generation_constants | item_dict
    logger.info("Generated map constants %r", result)
    return result


def split_bsp(
    bsp: Leaf,
    grid: npt.NDArray[np.int8],
    random_generator: random.Random,
    split_iteration: int,
) -> Leaf:
    """Split the bsp based on the generated constants.

    Parameters
    ----------
    bsp: Leaf
        The root leaf for the binary space partition.
    split_iteration: int
        The number of splits to perform.
    """
    # Start the splitting using deque
    deque_obj = deque["Leaf"]()
    deque_obj.append(bsp)
    while split_iteration and deque_obj:
        # Get the current leaf from the deque object
        current = deque_obj.popleft()

        # Split the bsp if possible
        if current.split(grid, random_generator) and current.left and current.right:
            # Add the child leafs so they can be split
            logger.debug("Split bsp. Split iteration is now %d", split_iteration)
            deque_obj.append(current.left)
            deque_obj.append(current.right)

            # Decrement the split iteration
            split_iteration -= 1

    # Return the splitted bsp
    return bsp


def generate_rooms(
    bsp: Leaf, grid: npt.NDArray[np.int8], random_generator: random.Random
) -> set[Rect]:
    """Generate the rooms for a given game level using the bsp.

    Parameters
    ----------
    bsp: Leaf
        The root leaf for the binary space partition.

    Returns
    -------
    set[Rect]
        The generated rooms.
    """
    # Create the rooms. We can use the same deque object since it is currently empty
    rooms: set[Rect] = set()
    deque_obj = deque["Leaf"]()
    deque_obj.append(bsp)
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
            while not current.create_room(grid, random_generator):
                # Width to height ratio is outside of range so try again
                logger.debug("Trying generation of room in leaf %r again", current)

            # Add the created room to the rooms list
            assert current.room  # Have to make sure its actually created first
            rooms.add(current.room)

    # Return all the created rooms
    return rooms


def create_mst(complete_graph: dict[Rect, list[Rect]]) -> set[tuple[float, Rect, Rect]]:
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


def create_hallways(
    grid: npt.NDArray[np.int8],
    random_generator: random.Random,
    connections: set[tuple[float, Rect, Rect]],
    obstacle_count: int,
) -> None:
    """Create the hallways by placing random obstacles and pathfinding around them.

    Parameters
    ----------
    grid: npt.NDArray[np.int8]
        The 2D grid which represents the dungeon.
    random_generator: random.Random
        The random generator used to pick the positions for the obstacles.
    connections: set[Rects]
        The connections to pathfind using the A* algorithm.
    obstacle_count: int
        The number of obstacles to place in the 2D grid.
    """
    # Place random obstacles in the grid
    obstacle_positions = [
        (count_y, count_x)
        for count_y, y in enumerate(grid)
        for count_x, x in enumerate(y)
        if x == TileType.EMPTY
    ]
    for _ in range(obstacle_count):
        grid[
            obstacle_positions.pop(random_generator.randrange(len(obstacle_positions)))
        ] = TileType.OBSTACLE
    logger.debug("Created %d obstacles in the 2D grid", obstacle_count)

    # Use the A* algorithm with to connect each pair of rooms making sure to avoid
    # the obstacles giving us natural looking hallways. Note that the width of the
    # hallways will always be odd in this implementation due to numpy indexing
    half_hallway_size = HALLWAY_SIZE // 2
    for _, pair_source, pair_destination in connections:
        for path_point_tup in calculate_astar_path(
            grid,
            pair_source.center,
            pair_destination.center,
        ):
            # Place a rect box around the path_point using HALLWAY_SIZE to determine
            # the width and height
            path_point = Point(*path_point_tup)
            logger.debug("Creating path from %r to %r", pair_source, pair_destination)
            Rect(
                Point(
                    path_point.x - half_hallway_size,
                    path_point.y - half_hallway_size,
                ),
                Point(
                    path_point.x + half_hallway_size,
                    path_point.y + half_hallway_size,
                ),
            ).place_rect(grid)


def place_tile(
    grid: npt.NDArray[np.int8],
    target_tile: TileType,
    possible_tiles: set[tuple[int, int]],
) -> None:
    """Places a given tile in the 2D grid.

    Parameters
    ----------
    grid: npt.NDArray[np.int8]
        The 2D grid which represents the dungeon.
    target_tile: TileType
        The tile to place in the 2D grid.
    possible_tiles: set[tuple[int, int]]
        The possible tiles that the tile can be placed into.
    """
    # Get a random floor position and place the target tile
    y, x = possible_tiles.pop()
    grid[y][x] = target_tile.value
    logger.debug("Placed tile %r in the 2D grid", target_tile)


def create_map(
    level: int, seed: int | str | None = None
) -> tuple[npt.NDArray[np.int8], LevelConstants]:
    """Generate the game map for a given game level.

    Parameters
    ----------
    level: int
        The game level to generate a map for.
    seed: int | str | None
        The seed to initialise the random generator. If it is None, then one will be
        generated.

    Raises
    ------
    ValueError
        Level must be bigger than or equal to 0.

    Returns
    -------
    tuple[npt.NDArray[np.int8], LevelConstants]
        The generated map and a named tuple containing the level, width and height.
    """
    # Check that the level number is valid
    if level < 0:
        raise ValueError("Level must be bigger than or equal to 0")

    # Create the random generator. If seed is None, then get a random 64-bit integer
    if seed is None:
        seed = random.getrandbits(64)
    random_generator = random.Random(seed)
    logger.debug("Generated state %r from seed %r", random_generator.getstate(), seed)

    # Initialise a few variables needed for the map generation
    map_constants = generate_constants(level)
    grid = np.full(
        (
            map_constants[GenerationConstantType.HEIGHT],
            map_constants[GenerationConstantType.WIDTH],
        ),
        TileType.EMPTY,
        TileType,
    )
    bsp = Leaf(
        Rect(
            Point(0, 0),
            Point(
                map_constants[GenerationConstantType.WIDTH] - 1,
                map_constants[GenerationConstantType.HEIGHT] - 1,
            ),
        ),
    )

    # Split the bsp and create the rooms
    rooms = generate_rooms(
        split_bsp(
            bsp,
            grid,
            random_generator,
            map_constants[GenerationConstantType.SPLIT_ITERATION],
        ),
        grid,
        random_generator,
    )

    # Create the hallways between the rooms
    complete_graph: dict[Rect, list[Rect]] = {}
    for source, destination in permutations(rooms, 2):
        complete_graph.update({source: complete_graph.get(source, []) + [destination]})
    create_hallways(
        grid,
        random_generator,
        create_mst(complete_graph),
        map_constants[GenerationConstantType.OBSTACLE_COUNT],
    )

    # Get all the tiles which can support items being placed on them
    possible_tiles: set[tuple[int, int]] = {
        (count_y, count_x)
        for count_y, y in enumerate(grid)
        for count_x, x in enumerate(y)
        if x == TileType.FLOOR
    }

    # Place the player tile and all the items tiles
    place_tile(grid, TileType.PLAYER, possible_tiles)
    for tile in ITEM_DISTRIBUTION:
        tiles_placed = 0
        while tiles_placed < map_constants[tile]:
            place_tile(grid, tile, possible_tiles)
            logger.debug("One of multiple %r placed in the 2D grid", tile)
            tiles_placed += 1

    # Return the map object and a GameMapShape object
    return grid, LevelConstants(level, grid.shape[1], grid.shape[0])
