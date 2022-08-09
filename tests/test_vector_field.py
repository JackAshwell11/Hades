"""Tests all functions in vector_field.py."""
from __future__ import annotations

# Pip
import arcade
import pytest

# Custom
from game.vector_field import VectorField

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
    temp_spritelist.extend([])
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
    return VectorField(50, 50, walls)


def test_vector_field_init(walls: arcade.SpriteList) -> None:
    """Test the initialisation of the VectorField class in vector_field.py.

    Parameters
    ----------
    walls: arcade.SpriteList
        The walls spritelist used for testing.
    """
    assert repr(VectorField(50, 50, walls)) == "<VectorField (Width=50) (Height=50)>"


def test_vector_field_get_tile_pos_for_pixel() -> None:
    """Test the get_tile_pos_for_pixel function in the VectorField class."""
    assert VectorField.get_tile_pos_for_pixel((500, 500)) == (8, 8)
    assert VectorField.get_tile_pos_for_pixel((0, 0)) == (0, 0)
    with pytest.raises(ValueError):
        VectorField.get_tile_pos_for_pixel((-500, -500))
    with pytest.raises(TypeError):
        VectorField.get_tile_pos_for_pixel(("test", "test"))  # noqa


def test_vector_field_recalculate_map(vector_field: VectorField) -> None:
    """Test the recalculate_map function in the VectorField class.

    Parameters
    ----------
    vector_field: VectorField
        The vector field used for testing.
    """


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
        vector_field.get_vector_direction(("test", "test"))  # noqa
