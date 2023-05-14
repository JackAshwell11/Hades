"""Manages the operations related to the sprite object."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import Sprite, Texture

from hades.constants import SPRITE_SCALE, GameObjectType

# Custom
from hades.textures import grid_pos_to_pixel

if TYPE_CHECKING:
    from hades.textures import TextureType

__all__ = ("HadesSprite",)


class HadesSprite(Sprite):
    """Represents a game object in the game.

    Attributes:
        textures_dict: The textures which represent this game object.
    """

    def __init__(
        self: HadesSprite,
        game_object_type: GameObjectType,
        position: tuple[float, float],
        texture_types: set[TextureType],
        *,
        blocking: bool,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_type: The type of the game object.
            position: The position of the game object on the screen.
            texture_types: The set of textures that relate to this sprite object.
            blocking: Whether this sprite blocks other game objects or not.
        """
        super().__init__(scale=SPRITE_SCALE)
        self.game_object_type: GameObjectType = game_object_type
        self.position = grid_pos_to_pixel(*position)
        self.textures_dict: dict[TextureType, Texture] = {
            texture: texture.value for texture in texture_types  # type: ignore[misc]
        }
        self.blocking: bool = blocking

        # Initialise the default sprite
        # TODO: Find proper way of doing this
        try:
            self.texture = list(self.textures_dict.values())[0]
        except TypeError:
            self.texture = list(self.textures_dict.values())[0][0]

    def __repr__(self: HadesSprite) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<HadesSprite (Game object type={self.game_object_type}) (Texture"
            f" count={len(self.textures_dict)}) (Blocking={self.blocking})>"
        )
