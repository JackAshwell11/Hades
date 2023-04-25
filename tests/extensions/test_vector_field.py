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


@pytest.fixture()
def walls() -> arcade.SpriteList:
    """Initialise the walls spritelist for use in testing.

    Returns
    -------
    arcade.SpriteList
        The walls spritelist for use in testing.
    """
    temp_spritelist = arcade.SpriteList()
    temp_spritelist.extend(
        [
            arcade.Sprite(center_x=84.0, center_y=84.0),
            arcade.Sprite(center_x=140.0, center_y=84.0),
            arcade.Sprite(center_x=196.0, center_y=84.0),
            arcade.Sprite(center_x=252.0, center_y=84.0),
            arcade.Sprite(center_x=308.0, center_y=84.0),
            arcade.Sprite(center_x=364.0, center_y=84.0),
            arcade.Sprite(center_x=420.0, center_y=84.0),
            arcade.Sprite(center_x=84.0, center_y=140.0),
            arcade.Sprite(center_x=420.0, center_y=140.0),
            arcade.Sprite(center_x=812.0, center_y=140.0),
            arcade.Sprite(center_x=868.0, center_y=140.0),
            arcade.Sprite(center_x=924.0, center_y=140.0),
            arcade.Sprite(center_x=980.0, center_y=140.0),
            arcade.Sprite(center_x=1036.0, center_y=140.0),
            arcade.Sprite(center_x=84.0, center_y=196.0),
            arcade.Sprite(center_x=420.0, center_y=196.0),
            arcade.Sprite(center_x=812.0, center_y=196.0),
            arcade.Sprite(center_x=1036.0, center_y=196.0),
            arcade.Sprite(center_x=84.0, center_y=252.0),
            arcade.Sprite(center_x=420.0, center_y=252.0),
            arcade.Sprite(center_x=812.0, center_y=252.0),
            arcade.Sprite(center_x=1036.0, center_y=252.0),
            arcade.Sprite(center_x=84.0, center_y=308.0),
            arcade.Sprite(center_x=420.0, center_y=308.0),
            arcade.Sprite(center_x=812.0, center_y=308.0),
            arcade.Sprite(center_x=1036.0, center_y=308.0),
            arcade.Sprite(center_x=84.0, center_y=364.0),
            arcade.Sprite(center_x=420.0, center_y=364.0),
            arcade.Sprite(center_x=812.0, center_y=364.0),
            arcade.Sprite(center_x=1036.0, center_y=364.0),
            arcade.Sprite(center_x=84.0, center_y=420.0),
            arcade.Sprite(center_x=420.0, center_y=420.0),
            arcade.Sprite(center_x=756.0, center_y=420.0),
            arcade.Sprite(center_x=812.0, center_y=420.0),
            arcade.Sprite(center_x=1036.0, center_y=420.0),
            arcade.Sprite(center_x=84.0, center_y=476.0),
            arcade.Sprite(center_x=420.0, center_y=476.0),
            arcade.Sprite(center_x=756.0, center_y=476.0),
            arcade.Sprite(center_x=980.0, center_y=476.0),
            arcade.Sprite(center_x=1036.0, center_y=476.0),
            arcade.Sprite(center_x=84.0, center_y=532.0),
            arcade.Sprite(center_x=420.0, center_y=532.0),
            arcade.Sprite(center_x=700.0, center_y=532.0),
            arcade.Sprite(center_x=756.0, center_y=532.0),
            arcade.Sprite(center_x=980.0, center_y=532.0),
            arcade.Sprite(center_x=84.0, center_y=588.0),
            arcade.Sprite(center_x=140.0, center_y=588.0),
            arcade.Sprite(center_x=364.0, center_y=588.0),
            arcade.Sprite(center_x=420.0, center_y=588.0),
            arcade.Sprite(center_x=700.0, center_y=588.0),
            arcade.Sprite(center_x=980.0, center_y=588.0),
            arcade.Sprite(center_x=140.0, center_y=644.0),
            arcade.Sprite(center_x=364.0, center_y=644.0),
            arcade.Sprite(center_x=588.0, center_y=644.0),
            arcade.Sprite(center_x=644.0, center_y=644.0),
            arcade.Sprite(center_x=700.0, center_y=644.0),
            arcade.Sprite(center_x=924.0, center_y=644.0),
            arcade.Sprite(center_x=980.0, center_y=644.0),
            arcade.Sprite(center_x=1036.0, center_y=644.0),
            arcade.Sprite(center_x=140.0, center_y=700.0),
            arcade.Sprite(center_x=364.0, center_y=700.0),
            arcade.Sprite(center_x=588.0, center_y=700.0),
            arcade.Sprite(center_x=1036.0, center_y=700.0),
            arcade.Sprite(center_x=28.0, center_y=756.0),
            arcade.Sprite(center_x=84.0, center_y=756.0),
            arcade.Sprite(center_x=140.0, center_y=756.0),
            arcade.Sprite(center_x=364.0, center_y=756.0),
            arcade.Sprite(center_x=420.0, center_y=756.0),
            arcade.Sprite(center_x=476.0, center_y=756.0),
            arcade.Sprite(center_x=532.0, center_y=756.0),
            arcade.Sprite(center_x=588.0, center_y=756.0),
            arcade.Sprite(center_x=1036.0, center_y=756.0),
            arcade.Sprite(center_x=28.0, center_y=812.0),
            arcade.Sprite(center_x=1036.0, center_y=812.0),
            arcade.Sprite(center_x=28.0, center_y=868.0),
            arcade.Sprite(center_x=1036.0, center_y=868.0),
            arcade.Sprite(center_x=28.0, center_y=924.0),
            arcade.Sprite(center_x=1036.0, center_y=924.0),
            arcade.Sprite(center_x=28.0, center_y=980.0),
            arcade.Sprite(center_x=420.0, center_y=980.0),
            arcade.Sprite(center_x=476.0, center_y=980.0),
            arcade.Sprite(center_x=532.0, center_y=980.0),
            arcade.Sprite(center_x=588.0, center_y=980.0),
            arcade.Sprite(center_x=1036.0, center_y=980.0),
            arcade.Sprite(center_x=28.0, center_y=1036.0),
            arcade.Sprite(center_x=84.0, center_y=1036.0),
            arcade.Sprite(center_x=140.0, center_y=1036.0),
            arcade.Sprite(center_x=196.0, center_y=1036.0),
            arcade.Sprite(center_x=252.0, center_y=1036.0),
            arcade.Sprite(center_x=308.0, center_y=1036.0),
            arcade.Sprite(center_x=364.0, center_y=1036.0),
            arcade.Sprite(center_x=420.0, center_y=1036.0),
            arcade.Sprite(center_x=588.0, center_y=1036.0),
            arcade.Sprite(center_x=644.0, center_y=1036.0),
            arcade.Sprite(center_x=700.0, center_y=1036.0),
            arcade.Sprite(center_x=756.0, center_y=1036.0),
            arcade.Sprite(center_x=812.0, center_y=1036.0),
            arcade.Sprite(center_x=868.0, center_y=1036.0),
            arcade.Sprite(center_x=924.0, center_y=1036.0),
            arcade.Sprite(center_x=980.0, center_y=1036.0),
            arcade.Sprite(center_x=1036.0, center_y=1036.0),
        ],
    )
    return temp_spritelist


@pytest.fixture()
def vector_field(walls: arcade.SpriteList) -> VectorField:
    """Initialise a vector field for use in testing.

    Parameters
    ----------
    walls: arcade.SpriteList
        The walls spritelist used for testing.

    Returns
    -------
    VectorField
        The vector field for use in testing.
    """
    return VectorField(walls, 20, 20)


def test_vector_field_init(walls: arcade.SpriteList) -> None:
    """Test the initialisation of the VectorField class in vector_field.py.

    Parameters
    ----------
    walls: arcade.SpriteList
        The walls spritelist used for testing.
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

    Parameters
    ----------
    vector_field: VectorField
        The vector field used for testing.
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
