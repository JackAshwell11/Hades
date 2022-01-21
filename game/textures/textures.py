from __future__ import annotations

import pathlib

# Builtin
from typing import Tuple

# Pip
import arcade

# Custom
from constants import SPRITE_HEIGHT, SPRITE_SCALE, SPRITE_WIDTH


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
        x * SPRITE_WIDTH * SPRITE_SCALE + SPRITE_WIDTH / 2 * SPRITE_SCALE,
        y * SPRITE_HEIGHT * SPRITE_SCALE + SPRITE_HEIGHT / 2 * SPRITE_SCALE,
    )


# Load the textures for each sprite
texture_path = pathlib.Path(__file__).resolve().parent.joinpath("images")
filenames = [
    "floor.png",
    "wall.png",
]
textures = [
    arcade.load_texture(str(texture_path.joinpath(filename))) for filename in filenames
]
