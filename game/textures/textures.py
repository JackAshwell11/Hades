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
        x * SPRITE_WIDTH + SPRITE_WIDTH / 2,
        y * SPRITE_HEIGHT + SPRITE_HEIGHT / 2,
    )


def calculate_max_camera_size(
    x: int, y: int, width: int, height: int
) -> Tuple[float, float]:
    """
    Calculates the max width and height the camera can be based on the grid size.

    Parameters
    ----------
    x: int
        The x index of the last item in the grid.
    y: int
        The y index of the last item in the grid.
    width: int
        The width of the viewport camera.
    height: int
        The height of the viewport camera.

    Returns
    -------
    Tuple[float, float]
        The max width and height that the camera can be.
    """
    upper_x, upper_y = calculate_position(x, y)
    return (
        upper_x - width + (width / SPRITE_WIDTH),
        upper_y - height + (height / SPRITE_HEIGHT),
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
