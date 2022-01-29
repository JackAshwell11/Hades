from __future__ import annotations

from enum import Enum

# Builtin
from typing import TYPE_CHECKING, Optional, Union

# Pip
import arcade

# Custom
from constants import SPRITE_SCALE
from textures.textures import pos_to_pixel, textures

if TYPE_CHECKING:
    from .enemy import Enemy
    from .player import Player
    from .tiles import Tile


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
    tile_id: TileType
        The tile ID for the entity.
    character: Optional[Union[Player, Enemy]]
        The character which this entity represents
    tile: Optional[Tile]
        The tile which this entity represents.
    item
        The item which this entity represents.

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
        tile_id: TileType,
        character: Optional[Union[Player, Enemy]] = None,
        tile: Optional[Tile] = None,
        item=None,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.texture: arcade.Texture = tile_id.value
        self.character: Optional[Union[Player, Enemy]] = character
        self.tile: Optional[Tile] = tile
        self.item = item

    def __repr__(self) -> str:
        return f"<Entity (Position=({self.center_x}, {self.center_y}))>"
