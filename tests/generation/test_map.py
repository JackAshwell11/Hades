"""Tests all functions in generation/map.py."""
from __future__ import annotations

# Builtin
from collections import deque
from typing import TYPE_CHECKING

# Pip
import numpy as np
import pytest

# Custom
from game.constants.generation import ENEMY_DISTRIBUTION, ITEM_DISTRIBUTION, TileType
from game.generation.map import GameMapShape, Map, create_map
from game.generation.primitives import Point

if TYPE_CHECKING:
    from game.generation.bsp import Leaf

__all__ = ()


def test_create_map() -> None:
    """Test the create_map function in map.py."""
    temp_positive = create_map(1)
    assert isinstance(temp_positive[0], np.ndarray) and isinstance(
        temp_positive[1], GameMapShape
    )
    temp_zero = create_map(0)
    assert isinstance(temp_zero[0], np.ndarray) and isinstance(
        temp_zero[1], GameMapShape
    )
    temp_negative = create_map(-1)
    assert isinstance(temp_negative[0], np.ndarray) and isinstance(
        temp_negative[1], GameMapShape
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
    assert (
        repr(Map(0))
        == "<Map (Width=30) (Height=20) (Split count=5) (Enemy count=7) (Item count=3)>"
    )


def test_map_place_multiple(map_obj: Map) -> None:
    """Test the _place_multiple function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """


def test_map_place_tile(map_obj: Map) -> None:
    """Test the _place_tile function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """


def test_map_private(map_obj: Map) -> None:
    """Test all the private functions in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    # Check if the _generate_constants function returns the necessary keys
    temp_result = set(map_obj._generate_constants().keys())
    assert temp_result == {
        "width",
        "height",
        "split iteration",
        "obstacle count",
        "enemy count",
        "item count",
    }.union(ENEMY_DISTRIBUTION.keys()).union(ITEM_DISTRIBUTION.keys())

    # Use a deque object to get all the leafs which don't have child leaves, so we can
    # check if the correct amount of splits has taken place (there should always be n+1
    # child leaves in the result list for this scenario)
    map_obj._split_bsp()
    result = []
    room_gen_deque = deque["Leaf"]()
    room_gen_deque.append(map_obj.bsp)
    while room_gen_deque:
        current: Leaf = room_gen_deque.popleft()
        if current.left and current.right:
            room_gen_deque.append(current.left)
            room_gen_deque.append(current.right)
        else:
            result.append(current)
    assert len(result) == map_obj.map_constants["split iteration"] + 1

    # Check if at least 1 room is generated
    rooms = map_obj._generate_rooms()
    assert rooms

    # Use a flood fill to check if all the rooms are connected
    map_obj._create_hallways(rooms)
    hallway_gen_deque = deque["Point"]()
    hallway_gen_deque.append(rooms[0].center)
    offsets: list[tuple[int, int]] = [
        (0, -1),
        (-1, 0),
        (1, 0),
        (0, 1),
    ]

    def bfs(target):
        height, width = map_obj.grid.shape
        for dx, dy in offsets:
            x, y = target.x + dx, target.y + dy

            if (0 <= x < width) and (0 <= y < height):
                yield Point(x, y)

    visited = set()
    centers = [room.center for room in rooms]
    centers_result = []
    while hallway_gen_deque:
        current: Point = hallway_gen_deque.popleft()
        map_obj.grid[current.y][current.x] = 9
        if current in centers:
            centers_result.append(True)
        for neighbour in bfs(current):
            if (
                map_obj.grid[neighbour.y][neighbour.x] == TileType.FLOOR
                and neighbour not in visited
            ):
                hallway_gen_deque.append(neighbour)
                visited.add(neighbour)
    print(len(centers_result) == len(centers))


def test_map_generate_map(map_obj: Map) -> None:
    """Test the _generate_map function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    temp_map = map_obj.generate_map()
    assert (
        isinstance(temp_map, Map)
        and temp_map.enemy_spawns
        and temp_map.player_pos != (-1, -1)
    )
