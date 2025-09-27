"""Manages the operations related to the sprite object."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import BasicSprite, Texture

# Custom
from hades_engine.ecs import SPRITE_SCALE, GameObjectType

if TYPE_CHECKING:
    from hades.constructors import GameObjectConstructor

__all__ = ("AnimatedSprite", "HadesSprite", "make_sprite")


class HadesSprite(BasicSprite):
    """Represents a sprite object in the game."""

    __slots__ = ("constructor", "game_object_id")

    def __init__(
        self: HadesSprite,
        game_object_id: int,
        constructor: GameObjectConstructor,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object's ID.
            constructor: The game object's constructor.
        """
        super().__init__(
            constructor.textures[0].get_texture(),
            SPRITE_SCALE,
        )
        self.game_object_id: int = game_object_id
        self.constructor: GameObjectConstructor = constructor
        self.depth = constructor.depth

    @property
    def game_object_type(self: HadesSprite) -> GameObjectType:
        """Return the game object's type.

        Returns:
            The game object's type.
        """
        return self.constructor.game_object_type

    @property
    def name(self: HadesSprite) -> str:
        """Return the game object's name.

        Returns:
            The game object's name.
        """
        return self.constructor.name

    @property
    def description(self: HadesSprite) -> str:
        """Return the game object's description.

        Returns:
            The game object's description.
        """
        return self.constructor.description


class AnimatedSprite(HadesSprite):
    """Represents an animated sprite object in the game.

    Attributes:
        sprite_textures: The sprite's textures.
    """

    __slots__ = ("sprite_textures",)

    def __init__(
        self: AnimatedSprite,
        game_object_id: int,
        constructor: GameObjectConstructor,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object's ID.
            constructor: The game object's constructor.
        """
        super().__init__(game_object_id, constructor)
        self.sprite_textures: list[tuple[Texture, Texture]] = [
            (texture.get_texture(), texture.get_texture().flip_left_right())
            for texture in constructor.textures
        ]


def make_sprite(game_object_id: int, constructor: GameObjectConstructor) -> HadesSprite:
    """Create a sprite object.

    Args:
        game_object_id: The game object's ID.
        constructor: The game object's constructor.

    Returns:
        The sprite object.
    """
    sprite_class = AnimatedSprite if len(constructor.textures) > 1 else HadesSprite
    return sprite_class(game_object_id, constructor)
