"""Tests all functions in generation/map.py."""
from __future__ import annotations

# Builtin
from collections import deque

# Pip
import numpy as np
import pytest

# Custom
from game.constants.generation import ENEMY_DISTRIBUTION, ITEM_DISTRIBUTION
from game.generation.map import GameMapShape, Map, create_map

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
        create_map("test")  # noqa


def test_game_map_shape() -> None:
    """Test the GameMapShape class in map.py."""
    temp_game_map_shape_one = GameMapShape(0, 0)
    assert temp_game_map_shape_one == (0, 0)
    temp_game_map_shape_two = GameMapShape("test", "test")  # noqa
    assert temp_game_map_shape_two == ("test", "test")


def test_map_init() -> None:
    """Test the initialisation of the Map class in map.py."""
    assert (
        repr(Map(0))
        == "<Map (Width=30) (Height=20) (Split count=5) (Enemy count=7) (Item count=3)>"
    )


def test_map_generate_constants(map_obj: Map) -> None:
    """Test the _generate_constants function in the Map class.

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


def test_map_split_bsp(map_obj: Map) -> None:
    """Test the _split_bsp function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
    # Use a deque object to get all the leafs which don't have child leaves
    map_obj._split_bsp()
    result = []
    deque_obj = deque["Leaf"]()
    deque_obj.append(map_obj.bsp)
    while deque_obj:
        current = deque_obj.popleft()
        if current.left and current.right:
            deque_obj.append(current.left)
            deque_obj.append(current.right)
        else:
            result.append(current)

    # There should now be n+1 child leaves in the result list
    assert len(result) == map_obj.map_constants["split iteration"] + 1


def test_map_generate_rooms(map_obj: Map) -> None:
    """Test the _generate_rooms function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """


def test_map_create_hallways(map_obj: Map) -> None:
    """Test the _create_hallways function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """


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


def test_map_generate_map(map_obj: Map) -> None:
    """Test the _generate_map function in the Map class.

    Parameters
    ----------
    map_obj: Map
        The map used for testing.
    """
