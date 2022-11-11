"""Tests all functions in generation/map.py."""
from __future__ import annotations

import logging
from collections import deque

# Builtin
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

# Pip
import pytest

# Custom
from hades.constants.generation import (
    ITEM_DISTRIBUTION,
    GenerationConstantType,
    TileType,
)
from hades.generation.map import (
    LevelConstants,
    add_extra_connections,
    create_hallways,
    create_map,
    create_mst,
    generate_constants,
    generate_rooms,
    place_tile,
    split_bsp,
)
from hades.generation.primitives import Point

if TYPE_CHECKING:
    from collections.abc import Generator

    from hades.generation.bsp import Leaf
    from hades.generation.primitives import Rect

__all__ = ()


def count_items(grid: np.ndarray) -> int:
    """Count the number of items generated in a given numpy grid.

    Parameters
    ----------
    grid: np.ndarray
        The numpy grid to count items in.

    Returns
    -------
    int
        The number of items in the grid.
    """
    return np.count_nonzero(np.isin(grid, list(ITEM_DISTRIBUTION.keys())))


def get_player_pos(grid: np.ndarray) -> tuple[int, int]:
    """Get the player position in a given 2D array.

    Parameters
    ----------
    grid: np.ndarray
        The numpy grid to find the player in.

    Returns
    -------
    tuple[int, int]
        The player position. If one is not found, this will be empty.
    """
    return np.where(grid == TileType.PLAYER)


def grid_bfs(
    target: tuple[int, int],
    height: int,
    width: int,
) -> Generator[tuple[int, int], None, None]:
    """Get a target's neighbours based on a given list of offsets.

    Note that this uses the same logic as the grid_bfs() function in the C++ extension,
    however, it is much slower due to Python.

    Parameters
    ----------
    target: tuple[int, int]
        The target to get neighbours for.
    height: int
        The height of the grid.
    width: int
        The width of the grid.

    Yields
    ------
    Generator[tuple[int, int], None, None]
        A list of the target's neighbours.
    """
    # Get all the neighbour floor tile positions relative to the current target
    for dx, dy in (
        (0, -1),
        (-1, 0),
        (1, 0),
        (0, 1),
    ):
        # Check if the neighbour position is within the boundaries of the grid or not
        x, y = target[0] + dx, target[1] + dy
        if (x < 0 or x >= width) or (y < 0 or y >= height):
            continue

        # Yield the neighbour tile position
        yield x, y


def get_possible_tiles(grid: np.ndarray) -> set[tuple[int, int]]:
    """Get the possible tiles that a game object can be placed into.

    Parameters
    ----------
    grid: np.ndarray
        The numpy grid to get possible tiles from.

    Returns
    -------
    set[tuple[int, int]]
        The set of possible tiles.
    """
    # This uses the same code from Map.generate_map()
    return set(zip(*np.nonzero(grid == TileType.FLOOR)))  # type: ignore


def test_create_map() -> None:
    """Test the create_map function in map.py."""
    temp_positive = create_map(1, 0)
    assert (
        isinstance(temp_positive[0], np.ndarray)
        and isinstance(temp_positive[1], LevelConstants)
        and count_items(temp_positive[0]) == 6
        and get_player_pos(temp_positive[0]) == (15, 21)
    )
    temp_zero = create_map(0, 0)
    assert (
        isinstance(temp_zero[0], np.ndarray)
        and isinstance(temp_zero[1], LevelConstants)
        and count_items(temp_zero[0]) == 5
        and get_player_pos(temp_zero[0]) == (15, 21)
    )
    temp_rand_seed = create_map(0)
    assert isinstance(temp_rand_seed[0], np.ndarray) and isinstance(
        temp_rand_seed[1], LevelConstants
    )
    with pytest.raises(ValueError):
        create_map(-1)
    with pytest.raises(TypeError):
        create_map("test")  # type: ignore


def test_level_constants() -> None:
    """Test the LevelConstants class in map.py."""
    assert LevelConstants(0, 0, 0) == (0, 0, 0)
    assert LevelConstants("test", "test", "test") == (  # type: ignore
        "test",
        "test",
        "test",
    )


def test_map_generate_constants() -> None:
    """Test the generate_constants function."""
    assert generate_constants(0) == {
        GenerationConstantType.WIDTH: 30,
        GenerationConstantType.HEIGHT: 20,
        GenerationConstantType.SPLIT_ITERATION: 5,
        GenerationConstantType.OBSTACLE_COUNT: 50,
        GenerationConstantType.ITEM_COUNT: 5,
        TileType.HEALTH_POTION: 2,
        TileType.ARMOUR_POTION: 2,
        TileType.HEALTH_BOOST_POTION: 1,
        TileType.ARMOUR_BOOST_POTION: 0,
        TileType.SPEED_BOOST_POTION: 0,
        TileType.FIRE_RATE_BOOST_POTION: 0,
    }


def test_map_split_bsp(
    caplog: pytest.LogCaptureFixture,
    leaf: Leaf,
    constants: dict[TileType | GenerationConstantType, int],
) -> None:
    """Test the split_bsp function.

    Parameters
    ----------
    caplog: pytest.LogCaptureFixture
        A fixture which allows logs to be captured
    leaf: Leaf
        The leaf used for testing.
    constants: dict[TileType | GenerationConstantType, int]
        The generated constants.
    """
    # Use a deque object to get all the leafs which don't have child leaves, so we can
    # check if the correct amount of splits has taken place (there should always be n+1
    # child leaves in the result list for this scenario)
    split_bsp(leaf, constants[GenerationConstantType.SPLIT_ITERATION])
    result = []
    room_gen_deque = deque["Leaf"]()
    room_gen_deque.append(leaf)
    while room_gen_deque:
        current_leaf: Leaf = room_gen_deque.popleft()
        if current_leaf.left and current_leaf.right:
            room_gen_deque.append(current_leaf.left)
            room_gen_deque.append(current_leaf.right)
        else:
            result.append(current_leaf)
    assert len(result) == constants[GenerationConstantType.SPLIT_ITERATION] + 1

    # Make sure we test what happens if the bsp is already split. To do this, we can
    # capture the logs and test that there are none
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    split_bsp(leaf, constants[GenerationConstantType.SPLIT_ITERATION])
    assert not caplog.records


def test_map_generate_rooms(
    leaf: Leaf, constants: dict[TileType | GenerationConstantType, int]
) -> None:
    """Test the generate_rooms function.

    Parameters
    ----------
    leaf: Leaf
        The leaf used for testing.
    constants: dict[TileType | GenerationConstantType, int]
        The generated constants.
    """
    # Check if at least 1 room is generated
    rooms = generate_rooms(
        split_bsp(leaf, constants[GenerationConstantType.SPLIT_ITERATION])
    )
    assert rooms
    assert not generate_rooms(leaf)


def test_map_create_hallways(
    leaf: Leaf, constants: dict[TileType | GenerationConstantType, int]
) -> None:
    """Test the create_hallways function.

    Parameters
    ----------
    leaf: Leaf
        The leaf used for testing.
    constants: dict[TileType | GenerationConstantType, int]
        The generated constants.
    """
    # Create the hallways using the same code fom map.py
    rooms = generate_rooms(
        split_bsp(leaf, constants[GenerationConstantType.SPLIT_ITERATION])
    )
    complete_graph: dict[Rect, list[Rect]] = {}
    for source, destination in permutations(rooms, 2):
        complete_graph.update({source: complete_graph.get(source, []) + [destination]})
    create_hallways(
        leaf.grid,
        leaf.random_generator,
        add_extra_connections(complete_graph, create_mst(complete_graph)),
        constants[GenerationConstantType.OBSTACLE_COUNT],
    )

    # Use a flood fill to check if all the rooms are connected
    hallway_gen_deque = deque["Point"]()
    hallway_gen_deque.append(rooms.pop().center)
    visited, reached = set(), set()
    centers = [room.center for room in rooms]
    while hallway_gen_deque:
        current_point: Point = hallway_gen_deque.popleft()
        if current_point in centers:
            reached.add(current_point)
        for bfs_neighbour in grid_bfs(current_point, *leaf.grid.shape):
            neighbour = Point(*bfs_neighbour)
            if (
                leaf.grid[neighbour.y][neighbour.x] is TileType.FLOOR
                and neighbour not in visited
            ):
                hallway_gen_deque.append(neighbour)
                visited.add(neighbour)
    assert len(reached) == len(centers)


def test_place_tile(
    leaf: Leaf, constants: dict[TileType | GenerationConstantType, int]
) -> None:
    """Test the place_tile function.

    Parameters
    ----------
    leaf: Leaf
        The leaf used for testing.
    constants: dict[TileType | GenerationConstantType, int]
        The generated constants.
    """
    # We only need to generate the rooms since we just want floor tiles
    generate_rooms(split_bsp(leaf, constants[GenerationConstantType.SPLIT_ITERATION]))
    possible_tiles = get_possible_tiles(leaf.grid)

    # Check if the player has been generated
    assert get_player_pos(leaf.grid) != (42, 2)
    place_tile(leaf.grid, TileType.PLAYER, possible_tiles)
    assert get_player_pos(leaf.grid) == (42, 2)

    # Check if an item has been generated
    assert count_items(leaf.grid) == 0
    place_tile(leaf.grid, TileType.HEALTH_POTION, possible_tiles)
    assert count_items(leaf.grid) == 1

    # Check if we have no possible tiles
    with pytest.raises(KeyError):
        place_tile(leaf.grid, TileType.PLAYER, set())
