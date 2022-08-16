"""Tests all functions in generation/map.py."""
from __future__ import annotations

# Builtin
from collections import deque
from typing import TYPE_CHECKING

# Pip
import numpy as np
import pytest

# Custom
from hades.common import grid_bfs
from hades.constants.generation import BASE_ITEM_COUNT, ITEM_DISTRIBUTION, TileType
from hades.generation.map import GameMapShape, Map, create_map
from hades.generation.primitives import Point

if TYPE_CHECKING:
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
    temp_positive = create_map(1)
    assert (
        isinstance(temp_positive[0].grid, np.ndarray)
        and isinstance(temp_positive[1], GameMapShape)
        and count_items(temp_positive[0].grid) == BASE_ITEM_COUNT
        and temp_positive[0].player_pos != (-1, -1)
    )
    temp_zero = create_map(0)
    assert (
        isinstance(temp_zero[0].grid, np.ndarray)
        and isinstance(temp_zero[1], GameMapShape)
        and count_items(temp_zero[0].grid) == BASE_ITEM_COUNT
        and temp_zero[0].player_pos != (-1, -1)
    )
    temp_negative = create_map(-1)
    assert (
        isinstance(temp_negative[0].grid, np.ndarray)
        and isinstance(temp_negative[1], GameMapShape)
        and count_items(temp_negative[0].grid) == BASE_ITEM_COUNT
        and temp_negative[0].player_pos != (-1, -1)
    )
    with pytest.raises(TypeError):
        create_map("test")  # type: ignore


def test_game_map_shape() -> None:
    """Test the GameMapShape class in map.py."""
    temp_game_map_shape_one = GameMapShape(0, 0)
    assert temp_game_map_shape_one == (0, 0)
    temp_game_map_shape_two = GameMapShape("test", "test")  # type: ignore
    assert temp_game_map_shape_two == ("test", "test")


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
        "width",
        "height",
        "split iteration",
        "obstacle count",
        "item count",
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
    assert len(result) == map_obj.map_constants["split iteration"] + 1

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
    assert count_items(map_obj.grid) == BASE_ITEM_COUNT
