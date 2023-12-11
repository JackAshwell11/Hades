"""Handles loading and storage of textures needed by the game."""

from __future__ import annotations

# Builtin
from enum import Enum
from pathlib import Path

# Pip
from arcade import Texture, load_texture, load_texture_pair

# Custom
from hades_extensions.game_objects import SPRITE_SIZE

__all__ = (
    "BiggerThanError",
    "TextureType",
    "grid_pos_to_pixel",
    "load_moving_texture",
    "load_non_moving_texture",
)


# Create the texture path
texture_path = Path(__file__).resolve().parent / "resources" / "textures"


class BiggerThanError(Exception):
    """Raised when a value is less than a required value."""

    def __init__(self: BiggerThanError, *, min_value: float) -> None:
        """Initialise the object.

        Args:
            min_value: The minimum value that is allowed.
        """
        super().__init__(f"The input must be bigger than or equal to {min_value}.")


def load_moving_texture(texture: str) -> tuple[Texture, Texture]:
    """Load a moving texture into the texture cache.

    Args:
        texture: The moving texture to load

    Returns:
        The loaded moving texture.
    """
    return load_texture_pair(texture_path.joinpath(texture))


def load_non_moving_texture(texture: str) -> Texture:
    """Load a non-moving texture into the texture cache.

    Args:
        texture: The non-moving texture to load

    Returns:
        The loaded non-moving texture.
    """
    return load_texture(texture_path.joinpath(texture))


class TextureType(Enum):
    """Stores the different types of textures that exist."""

    ARMOUR_BOOST_POTION = load_non_moving_texture("armour_boost_potion.png")
    ARMOUR_POTION = load_non_moving_texture("armour_potion.png")
    ENEMY_IDLE = load_moving_texture("enemy_idle.png")
    FIRE_RATE_BOOST_POTION = load_non_moving_texture("fire_rate_boost_potion.png")
    FLOOR = load_non_moving_texture("floor.png")
    HEALTH_BOOST_POTION = load_non_moving_texture("health_boost_potion.png")
    HEALTH_POTION = load_non_moving_texture("health_potion.png")
    PLAYER_IDLE = load_moving_texture("player_idle.png")
    SHOP = load_non_moving_texture("shop.png")
    SPEED_BOOST_POTION = load_non_moving_texture("speed_boost_potion.png")
    WALL = load_non_moving_texture("wall.png")


def grid_pos_to_pixel(x: int, y: int) -> tuple[float, float]:
    """Calculate the x and y position based on the dungeon or vector field position.

    Args:
        x: The x position in the dungeon or vector field.
        y: The x position in the dungeon or vector field.

    Returns:
        The x and y position of a sprite on the screen.

    Raises:
        BiggerThanError: The input must be bigger than or equal to 0.
    """
    # Check if the inputs are negative
    if x < 0 or y < 0:
        raise BiggerThanError(min_value=0)

    # Calculate the position on screen
    return (
        x * SPRITE_SIZE + SPRITE_SIZE / 2,
        y * SPRITE_SIZE + SPRITE_SIZE / 2,
    )
