from __future__ import annotations

# Pip
import arcade

# Custom
from textures.textures import textures


class Tile(arcade.Sprite):
    def __init__(self, x, y, tile_type):
        super().__init__()
        self.texture = textures[tile_type - 1]
        self.center_x = x * 16 * 1 + 16 / 2 * 1
        self.center_y = y * 16 * 1 + 16 / 2 * 1
