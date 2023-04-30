"""Tests all functions in extensions/vector_field.py."""
from __future__ import annotations

# Builtin
import random

# Pip
import arcade
import pytest

# Custom
from hades.extensions import VectorField

__all__ = ()


def test_vector_field_init(walls: arcade.SpriteList) -> None:
    """Test the initialisation of the VectorField class in vector_field.py.

    Args:
        walls: The walls spritelist used for testing.
    """
    assert repr(VectorField(walls, 20, 20)) == "<VectorField (Width=20) (Height=20)>"


def test_vector_field_get_tile_pos_for_pixel() -> None:
    """Test the get_tile_pos_for_pixel function in the VectorField class."""
    assert VectorField.pixel_to_grid_pos((500, 500)) == (8, 8)
    assert VectorField.pixel_to_grid_pos((0, 0)) == (0, 0)
    with pytest.raises(SystemError):
        VectorField.pixel_to_grid_pos((-500, -500))
    with pytest.raises(SystemError):
        VectorField.pixel_to_grid_pos(("test", "test"))  # type: ignore


def test_vector_field_recalculate_map(vector_field: VectorField) -> None:
    """Test the recalculate_map function in the VectorField class.

    Args:
        vector_field: The vector field used for testing.
    """
    # Pick 20 random positions and follow the vector dict checking if they reach the
    # player origin
    vector_result = []
    player_screen_pos = 252.0, 812.0
    player_grid_pos = vector_field.pixel_to_grid_pos(player_screen_pos)
    player_view_distance = 5
    temp_possible_spawns_valid = vector_field.recalculate_map(
        player_screen_pos,
        player_view_distance,
    )
    vector_dict_items = list(vector_field.vector_dict.items())
    for current_pos, vector in random.choices(vector_dict_items, k=20):
        while current_pos != player_grid_pos:
            # We need to check if an error has occurred
            if vector == (0, 0):
                vector_result.append(False)
                break

            # Trace back through the vector dict
            current_pos = current_pos[0] + vector[0], current_pos[1] + vector[1]
            vector = vector_field.vector_dict[current_pos]
        vector_result.append(True)
    assert all(vector_result)

    # Check that all possible spawns are 5 tiles away from the player
    for spawn in temp_possible_spawns_valid:
        # Note that this is the exact same logic as the Manhattan heuristic. This is
        # because the Dijkstra map technically calculates the Manhattan distance for
        # every tile from the origin tile
        distance = abs(player_grid_pos[0] - spawn[0]) + abs(
            player_grid_pos[1] - spawn[1],
        )
        assert distance > player_view_distance

    # Make sure we test what happens if the player's view distance is zero or negative
    with pytest.raises(SystemError):
        vector_field.recalculate_map(player_screen_pos, 0)
    with pytest.raises(SystemError):
        vector_field.recalculate_map(player_screen_pos, -1)
