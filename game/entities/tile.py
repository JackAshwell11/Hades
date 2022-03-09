from __future__ import annotations

# Pip
import arcade

# Custom
from constants import SPRITE_SCALE
from textures import pos_to_pixel


class Tile(arcade.Sprite):
    """
    Represents a tile in the game.

    Parameters
    ----------
    x: int
        The x position of the tile in the game map.
    y: int
        The y position of the tile in the game map.
    texture: arcade.Texture
        The sprite which represents this tile.

    Attributes
    ----------
    center_x: float
        The x position of the tile on the screen.
    center_y: float
        The y position of the tile on the screen.
    """

    def __init__(
        self,
        x: int,
        y: int,
        texture: arcade.Texture,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.texture: arcade.Texture = texture

    def __repr__(self) -> str:
        return f"<Tile (Position=({self.center_x}, {self.center_y}))>"
