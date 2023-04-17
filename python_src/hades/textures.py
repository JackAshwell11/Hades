"""Pre-loads all the textures needed by the game and stores them for later use."""
from __future__ import annotations

# Builtin
import logging
from enum import Enum
from pathlib import Path

# Pip
import arcade

# Custom
from hades.constants.game_objects import SPRITE_SIZE
from hades.exceptions import BiggerThanError

__all__ = (
    "MovingTextureType",
    "NonMovingTextureType",
    "grid_pos_to_pixel",
    "moving_textures",
    "non_moving_textures",
)

# Get the logger
logger = logging.getLogger(__name__)


class NonMovingTextureType(Enum):
    """Stores the different types of non-moving textures that exist."""

    FLOOR = "floor.png"
    WALL = "wall.png"
    HEALTH_POTION = "health_potion.png"
    ARMOUR_POTION = "armour_potion.png"
    HEALTH_BOOST_POTION = "health_boost_potion.png"
    ARMOUR_BOOST_POTION = "armour_boost_potion.png"
    SPEED_BOOST_POTION = "speed_boost_potion.png"
    FIRE_RATE_BOOST_POTION = "fire_rate_boost_potion.png"
    SHOP = "shop.png"


class MovingTextureType(Enum):
    """Stores the different types of moving textures that exist."""

    PLAYER_IDLE = "player_idle.png"
    ENEMY_IDLE = "enemy_idle.png"


def grid_pos_to_pixel(x: int, y: int) -> tuple[float, float]:
    """Calculate the x and y position based on the game map or vector field position.

    Parameters
    ----------
    x: int
        The x position in the game map or vector field.
    y: int
        The x position in the game map or vector field.

    Raises
    ------
    BiggerThanError
        The input must be bigger than or equal to 0.

    Returns
    -------
    tuple[float, float]
        The x and y position of a sprite on the screen.
    """
    # Check if the inputs are negative
    if x < 0 or y < 0:
        raise BiggerThanError(0)

    # Calculate the position on screen
    return (
        x * SPRITE_SIZE + SPRITE_SIZE / 2,
        y * SPRITE_SIZE + SPRITE_SIZE / 2,
    )


# Create the texture path
texture_path = Path(__file__).resolve().parent / "resources" / "textures"

# Create the non-moving textures
non_moving_textures: dict[NonMovingTextureType, arcade.Texture] = {
    non_moving_type: arcade.load_texture(texture_path.joinpath(non_moving_type.value))
    for non_moving_type in NonMovingTextureType
}

# Create the moving textures
moving_textures: dict[MovingTextureType, list[arcade.Texture]] = {
    moving_type: arcade.load_texture_pair(texture_path.joinpath(moving_type.value))
    for moving_type in MovingTextureType
}
