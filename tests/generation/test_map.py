"""Tests all functions in generation/map.py."""
from __future__ import annotations

# Builtin
from collections import deque
from typing import TYPE_CHECKING

# Pip
import numpy as np
import pytest

# Custom
from hades.constants.generation import (
    ITEM_DISTRIBUTION,
    MAP_GENERATION_COUNTS,
    GenerationConstantType,
    TileType,
)
from hades.generation.map import LevelConstants, Map, create_map
from hades.generation.primitives import Point

if TYPE_CHECKING:
    from collections.abc import Generator

    from hades.generation.bsp import Leaf

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

    Returns
    -------
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


def get_possible_tiles(grid: np.ndarray) -> list[tuple[int, int]]:
    """Get the possible tiles that a game object can be placed into.

    Parameters
    ----------
    grid: np.ndarray
        The numpy grid to get possible tiles from.

    Returns
    -------
    list[tuple[int, int]]
        The list of possible tiles.
    """
    # This uses the same code from Map.generate_map()
    return list(zip(*np.nonzero(grid == TileType.FLOOR)))  # noqa


def test_create_map() -> None:
    """Test the create_map function in map.py."""
    base_item_count = MAP_GENERATION_COUNTS[
        GenerationConstantType.ITEM_COUNT
    ].base_value
    temp_positive = create_map(1)
    assert (
        isinstance(temp_positive[0].grid, np.ndarray)
        and isinstance(temp_positive[1], LevelConstants)
        and count_items(temp_positive[0].grid) == base_item_count
        and temp_positive[0].player_pos != (-1, -1)
    )
    temp_zero = create_map(0)
    assert (
        isinstance(temp_zero[0].grid, np.ndarray)
        and isinstance(temp_zero[1], LevelConstants)
        and count_items(temp_zero[0].grid) == base_item_count
        and temp_zero[0].player_pos != (-1, -1)
    )
    temp_negative = create_map(-1)
    assert (
        isinstance(temp_negative[0].grid, np.ndarray)
        and isinstance(temp_negative[1], LevelConstants)
        and count_items(temp_negative[0].grid) == base_item_count
        and temp_negative[0].player_pos != (-1, -1)
    )
    with pytest.raises(TypeError):
        create_map("test")  # type: ignore


def test_level_constants() -> None:
    """Test the LevelConstants class in map.py."""
    temp_level_constants_one = LevelConstants(0, 0, 0)
    assert temp_level_constants_one == (0, 0, 0)
    temp_level_constants_two = LevelConstants("test", "test", "test")  # type: ignore
    assert temp_level_constants_two == ("test", "test", "test")


def test_map_init() -> None:
    """Test the initialisation of the Map class in map.py."""
    assert repr(Map(0)) == "<Map (Width=30) (Height=20) (Split count=5) (Item count=3)>"


def test_map_properties(map_obj: Map) -> None:
    """Test all the properties in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    assert map_obj.width == 30 and map_obj.height == 20


def test_map_generate_constants(map_obj: Map) -> None:
    """Test the generate_constants function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    temp_result = set(map_obj.generate_constants().keys())
    assert temp_result == {
        GenerationConstantType.WIDTH,
        GenerationConstantType.HEIGHT,
        GenerationConstantType.SPLIT_ITERATION,
        GenerationConstantType.OBSTACLE_COUNT,
        GenerationConstantType.ITEM_COUNT,
    }.union(ITEM_DISTRIBUTION.keys())


def test_map_split_bsp(map_obj: Map) -> None:
    """Test the split_bsp function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    # Use a deque object to get all the leafs which don't have child leaves, so we can
    # check if the correct amount of splits has taken place (there should always be n+1
    # child leaves in the result list for this scenario)
    map_obj.split_bsp()
    result = []
    room_gen_deque = deque["Leaf"]()
    room_gen_deque.append(map_obj.bsp)
    while room_gen_deque:
        current_leaf: Leaf = room_gen_deque.popleft()
        if current_leaf.left and current_leaf.right:
            room_gen_deque.append(current_leaf.left)
            room_gen_deque.append(current_leaf.right)
        else:
            result.append(current_leaf)
    assert (
        len(result) == map_obj.map_constants[GenerationConstantType.SPLIT_ITERATION] + 1
    )

    # Make sure we test what happens if the bsp is already split
    assert isinstance(map_obj.split_bsp(), Map)


def test_map_generate_rooms(map_obj: Map) -> None:
    """Test the generate_rooms function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    # Check if at least 1 room is generated
    rooms = map_obj.split_bsp().generate_rooms()
    assert rooms
    assert not map_obj.generate_rooms()


def test_map_create_hallways(map_obj: Map) -> None:
    """Test the create_hallways function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    # Use a flood fill to check if all the rooms are connected
    rooms = map_obj.split_bsp().generate_rooms()
    map_obj.create_hallways(rooms)
    hallway_gen_deque = deque["Point"]()
    hallway_gen_deque.append(rooms[0].center)
    visited, reached = set(), set()
    centers = [room.center for room in rooms]
    while hallway_gen_deque:
        current_point: Point = hallway_gen_deque.popleft()
        if current_point in centers:
            reached.add(current_point)
        for bfs_neighbour in grid_bfs(current_point, *map_obj.grid.shape):
            neighbour = Point(*bfs_neighbour)
            if (
                map_obj.grid[neighbour.y][neighbour.x] is TileType.FLOOR
                and neighbour not in visited
            ):
                hallway_gen_deque.append(neighbour)
                visited.add(neighbour)
    assert len(reached) == len(centers)


def test_map_place_tile(map_obj: Map) -> None:
    """Test the place_tile function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    # Make sure the dungeon is generated first
    rooms = map_obj.split_bsp().generate_rooms()
    map_obj.create_hallways(rooms)
    possible_tiles = get_possible_tiles(map_obj.grid)
    np.random.shuffle(possible_tiles)
    map_obj.place_tile(TileType.PLAYER, [])
    assert map_obj.player_pos == (-1, -1)
    map_obj.place_tile(TileType.PLAYER, possible_tiles)
    assert map_obj.player_pos != (-1, -1)


def test_map_place_multiple(map_obj: Map) -> None:
    """Test the place_multiple function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    # Make sure the dungeon is generated first
    rooms = map_obj.split_bsp().generate_rooms()
    map_obj.create_hallways(rooms)
    possible_tiles = get_possible_tiles(map_obj.grid)
    np.random.shuffle(possible_tiles)
    map_obj.place_multiple(ITEM_DISTRIBUTION, [])
    assert count_items(map_obj.grid) == 0
    map_obj.place_multiple(ITEM_DISTRIBUTION, possible_tiles)
    assert (
        count_items(map_obj.grid)
        == MAP_GENERATION_COUNTS[GenerationConstantType.ITEM_COUNT].base_value
    )
