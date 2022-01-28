from __future__ import annotations

# Pip
import arcade
from constants import SPRITE_SCALE

# Custom
from textures.textures import calculate_position, enemy_textures


class Enemy(arcade.Sprite):
    """
    Represents an enemy character in the game.

    Parameters
    ----------
    x: int
        The x position of the enemy in the game map.
    y: int
        The y position of the enemy in the game map.

    Attributes
    ----------
    center_x: float
        The x position of the enemy on the screen.
    center_y: float
        The y position of the enemy on the screen.
    """

    def __init__(self, x: int, y: int) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.texture: arcade.Texture = enemy_textures[0]
        self.center_x, self.center_y = calculate_position(x, y)

    def __repr__(self) -> str:
        return f"<Enemy (Position=({self.center_x}, {self.center_y}))>"
