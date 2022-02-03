from __future__ import annotations

# Builtin
import pathlib
from typing import Tuple

# Pip
import arcade

# Custom
from constants import SPRITE_HEIGHT, SPRITE_WIDTH


def pos_to_pixel(x: int, y: int) -> Tuple[float, float]:
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


# Create the texture path
texture_path = pathlib.Path(__file__).resolve().parent.joinpath("images")

# Create a list to hold all the filenames for the textures
filenames = {
    "tiles": [
        "floor.png",
        "wall.png",
    ],
    "player": [
        "player.png",
    ],
    "enemy": [
        "enemy.png",
    ],
    "attack": ["bullet.png"],
}

# Create the textures
textures = {}
for key, value in filenames.items():
    textures[key] = [
        arcade.load_texture(str(texture_path.joinpath(filename))) for filename in value
    ]
