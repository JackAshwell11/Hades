from __future__ import annotations

# Pip
import arcade

# Custom
from entities.base import Tile
from textures import non_moving_textures


class Floor(Tile):
    """
    Represents a floor tile in the game.

    Parameters
    ----------
    x: int
        The x position of the floor tile in the game map.
    y: int
        The y position of the floor tile in the game map.
    """

    # Class variables
    raw_texture: arcade.Texture | None = non_moving_textures["tiles"][0]

    def __init__(
        self,
        x: int,
        y: int,
    ) -> None:
        super().__init__(x, y)

    def __repr__(self) -> str:
        return f"<Floor (Position=({self.center_x}, {self.center_y}))>"


class Wall(Tile):
    """
    Represents a wall tile in the game.

    Parameters
    ----------
    x: int
        The x position of the wall tile in the game map.
    y: int
        The y position of the wall tile in the game map.
    """

    # Class variables
    raw_texture: arcade.Texture | None = non_moving_textures["tiles"][1]

    def __init__(
        self,
        x: int,
        y: int,
    ) -> None:
        super().__init__(x, y)

    def __repr__(self) -> str:
        return f"<Wall (Position=({self.center_x}, {self.center_y}))>"
