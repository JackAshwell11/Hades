from __future__ import annotations

# Builtin
import pathlib
from typing import Tuple

# Pip
import arcade

# Custom
from constants import SPRITE_HEIGHT, SPRITE_WIDTH


def calculate_position(x: int, y: int) -> Tuple[float, float]:
    """
    Calculate the x and y position based on the game map position.

    Parameters
    ----------
    x: int
        The x position in the game map.
    y: int
        The x position in the game map.

    Returns
    -------
    Tuple[float, float]
        The x and y position of a sprite on the screen.
    """
    return (
        x * SPRITE_WIDTH * 1 + SPRITE_WIDTH / 2 * 1,
        y * SPRITE_HEIGHT * 1 + SPRITE_HEIGHT / 2 * 1,
    )


# Create the texture path
texture_path = pathlib.Path(__file__).resolve().parent.joinpath("images")

# Create the tile textures
tile_filenames = [
    "floor.png",
    "wall.png",
]
tile_textures = [
    arcade.load_texture(str(texture_path.joinpath(filename)))
    for filename in tile_filenames
]

# Create the player textures
player_filename = ["player.png"]
player_textures = [
    arcade.load_texture(str(texture_path.joinpath(filename)))
    for filename in player_filename
]
