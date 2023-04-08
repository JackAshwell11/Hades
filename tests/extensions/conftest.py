"""Holds fixtures and test data used by extensions/."""
from __future__ import annotations

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
