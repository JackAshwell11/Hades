"""Tests all functions in vector_field.py."""
from __future__ import annotations

# Pip
import arcade
import numpy as np
import pytest

# Custom
from hades.vector_field import VectorField

__all__ = ()


@pytest.fixture
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
        ]
    )
    return temp_spritelist


@pytest.fixture
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
    return VectorField(20, 20, walls)


def test_vector_field_init(walls: arcade.SpriteList) -> None:
    """Test the initialisation of the VectorField class in vector_field.py.

    Parameters
    ----------
    walls: arcade.SpriteList
        The walls spritelist used for testing.
    """
    assert repr(VectorField(20, 20, walls)) == "<VectorField (Width=20) (Height=20)>"


def test_vector_field_get_tile_pos_for_pixel() -> None:
    """Test the get_tile_pos_for_pixel function in the VectorField class."""
    assert VectorField.get_tile_pos_for_pixel((500, 500)) == (8, 8)
    assert VectorField.get_tile_pos_for_pixel((0, 0)) == (0, 0)
    with pytest.raises(ValueError):
        VectorField.get_tile_pos_for_pixel((-500, -500))
    with pytest.raises(TypeError):
        VectorField.get_tile_pos_for_pixel(("test", "test"))  # type: ignore


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
    player_grid_pos = vector_field.get_tile_pos_for_pixel(player_screen_pos)
    vector_field.recalculate_map(player_screen_pos)
    vector_dict_items = list(vector_field.vector_dict.items())
    for index in np.random.choice(len(vector_dict_items), 20):
        current_pos, vector = vector_dict_items[index]
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


def test_vector_field_get_vector_direction(vector_field: VectorField) -> None:
    """Test the get_vector_direction function in the VectorField class.

    Parameters
    ----------
    vector_field: VectorField
        The vector field used for testing.
    """
    vector_field.vector_dict[(8, 8)] = (-1, -1)
    assert vector_field.get_vector_direction((500, 500)) == (-1, -1)
    with pytest.raises(KeyError):
        vector_field.get_vector_direction((0, 0))
    with pytest.raises(TypeError):
        vector_field.get_vector_direction(("test", "test"))  # type: ignore
