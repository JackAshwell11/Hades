from __future__ import annotations

# Builtin
from enum import Enum
from typing import TYPE_CHECKING, Optional

# Pip
import arcade

# Custom
from constants import SPRITE_SCALE
from textures.textures import pos_to_pixel, textures

if TYPE_CHECKING:
    from .character import Character


class TileType(Enum):
    FLOOR = textures["tiles"][0]
    WALL = textures["tiles"][1]
    PLAYER = textures["player"][0]
    ENEMY = textures["enemy"][0]


class Entity(arcade.Sprite):
    """
    Represents an on-screen sprite in the game.

    Parameters
    ----------
    x: int
        The x position of the sprite in the game map.
    y: int
        The y position of the sprite in the game map.
    tile_type: TileType
        The tile type for the entity.
    character: Optional[Character]
        The character which this entity represents
    is_tile: bool
        Whether the entity is a tile or not.
    item
        The item which this entity represents.
    ai
        The AI algorithm which this entity uses.

    Attributes
    ----------
    texture: arcade.Texture
        The sprite which represents this entity.
    center_x: float
        The x position of the sprite on the screen.
    center_y: float
        The y position of the sprite on the screen.-
    """

    def __init__(
        self,
        x: int,
        y: int,
        tile_type: TileType,
        character: Optional[Character] = None,
        is_tile: bool = False,
        item=None,
        ai=None,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.texture: arcade.Texture = tile_type.value
        self.character: Optional[Character] = character
        self.tile: bool = is_tile
        self.item = item
        self.ai = ai

    def __repr__(self) -> str:
        return f"<Entity (Position=({self.center_x}, {self.center_y}))>"
