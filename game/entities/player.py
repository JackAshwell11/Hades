from __future__ import annotations

# Pip
import arcade

# Custom
from textures.textures import calculate_position, player_textures


class Player(arcade.Sprite):
    """
    Represents a playable character in the game.

    Parameters
    ----------
    x: int
        The x position of the player in the game map.
    y: int
        The y position of the player in the game map.

    Attributes
    ----------
    center_x: float
        The x position of the player on the screen.
    center_y: float
        The y position of the player on the screen.
    """

    def __init__(self, x: int, y: int) -> None:
        super().__init__()
        self.texture: arcade.Texture = player_textures[0]
        self.center_x, self.center_y = calculate_position(x, y)
