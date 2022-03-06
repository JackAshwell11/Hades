from __future__ import annotations

# Builtin
import pathlib
from typing import Dict, List, Tuple

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
texture_path = (
    pathlib.Path(__file__).resolve().parent.joinpath("resources").joinpath("images")
)

# Create a dictionary to hold all the filenames for the non-moving textures
non_moving_filenames = {
    "tiles": [
        "floor.png",
        "wall.png",
    ],
    "attack": ["bullet.png", "cone.png"],
}

# Create a dictionary to hold all the filenames for the non-moving textures
moving_filenames = {
    "player": {
        "idle": ["player_idle.png"],
    },
    "enemy": {
        "idle": ["enemy_idle.png"],
    },
}

# Create the non-moving textures
non_moving_textures: Dict[str, List[arcade.Texture]] = {
    key: [arcade.load_texture(texture_path.joinpath(filename)) for filename in value]
    for key, value in non_moving_filenames.items()
}

# Create the moving textures
moving_textures: Dict[str, Dict[str, List[List[arcade.Texture]]]] = {
    key: {
        animation_type: [
            arcade.load_texture_pair(texture_path.joinpath(filename))
            for filename in sublist
        ]
        for animation_type, sublist in value.items()
    }
    for key, value in moving_filenames.items()
}
