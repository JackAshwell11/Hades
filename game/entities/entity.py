from __future__ import annotations

# Builtin
from enum import Enum
from typing import TYPE_CHECKING, Optional

# Pip
import arcade

# Custom
from constants import SPRITE_SCALE, SPRITE_WIDTH
from textures.textures import pos_to_pixel, textures

if TYPE_CHECKING:
    from entities.character import Character
    from physics import PhysicsEngine


class TileType(Enum):
    """Stores the references to the textures for each tile."""

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

    Attributes
    ----------
    texture: arcade.Texture
        The sprite which represents this entity.
    center_x: float
        The x position of the sprite on the screen.
    center_y: float
        The y position of the sprite on the screen.
    direction: float
        The angle the entity is facing.
    facing: int
        The direction the entity is facing. 0 is right and 1 is left.
    cone: arcade.Sprite
        The cone sprite which represents the attack range of this entity.
    """

    def __init__(
        self,
        x: int,
        y: int,
        tile_type: TileType,
        character: Optional[Character] = None,
        is_tile: bool = False,
        item=None,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.character: Optional[Character] = character
        self.tile: bool = is_tile
        self.item = item
        self.texture: arcade.Texture = tile_type.value
        self.direction: float = 0
        self.facing: int = 0
        self.cone: arcade.Sprite = arcade.Sprite(
            scale=SPRITE_SCALE,
            center_x=self.center_x + (SPRITE_WIDTH / 2),
            center_y=self.center_y,
            texture=textures["attack"][1],
        )

        # Set a few internal variables to allow various things to work
        if self.character:
            self.character.owner = self
            if self.character.ai:
                self.character.ai.owner = self

    def __repr__(self) -> str:
        return f"<Entity (Position=({self.center_x}, {self.center_y}))>"

    def pymunk_moved(
        self, physics_engine: PhysicsEngine, dx: float, dy: float, d_angle: float
    ) -> None:
        """
        Called by the pymunk physics engine if this entity moves.
        Parameters
        ----------
        physics_engine: PhysicsEngine
            The pymunk physics engine which manages physics for this entity.
        dx: float
            The change in x. Positive means the entity is travelling right and negative
            means left.
        dy: float
            The change in y. Positive means the entity is travelling up and negative
            means down.
        d_angle: float
            The change in the angle. This shouldn't be used in this game.
        """
        # Move the cone
        self.cone.center_x += dx
        self.cone.center_y += dy
