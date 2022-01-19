from __future__ import annotations

import pathlib

# Pip
import arcade

texture_path = pathlib.Path(__file__).resolve().parent.joinpath("images")

filenames = [
    "floor.png",
    "wall.png",
]

textures = [
    arcade.load_texture(str(texture_path.joinpath(filename))) for filename in filenames
]
