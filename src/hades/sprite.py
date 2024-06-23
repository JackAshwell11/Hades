"""Manages the operations related to the sprite object."""

from __future__ import annotations

# Pip
from arcade import (
    BasicSprite,
    Texture,
    color,
    make_soft_square_texture,
)

# Custom
from hades_extensions.game_objects import (
    SPRITE_SCALE,
    GameObjectType,
    Registry,
    Vec2d,
    grid_pos_to_pixel,
)
from hades_extensions.game_objects.components import KinematicComponent

__all__ = (
    "AnimatedSprite",
    "Bullet",
    "HadesSprite",
)


class HadesSprite(BasicSprite):
    """Represents a sprite object in the game.

    Attributes:
        game_object_id: The game object's ID.
        game_object_type: The game object's type.
    """

    __slots__ = ("game_object_id", "game_object_type", "registry")

    def __init__(
        self: HadesSprite,
        registry: Registry,
        game_object: tuple[int, GameObjectType],
        position: Vec2d,
        sprite_textures: list[Texture],
    ) -> None:
        """Initialise the object.

        Args:
            registry: The registry that manages the game objects, components, and
            systems.
            game_object: The game object's ID and type.
            position: The position of the sprite object in the grid.
            sprite_textures: The sprites' texture paths.
        """
        super().__init__(
            sprite_textures[0],
            SPRITE_SCALE,
            *grid_pos_to_pixel(position),
        )
        self.game_object_id, self.game_object_type = game_object
        self.registry: Registry = registry

    def update(self: HadesSprite) -> None:
        """Update the sprite object."""
        self.position = self.registry.get_component(
            self.game_object_id,
            KinematicComponent,
        ).get_position()

    def __repr__(self: HadesSprite) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<HadesSprite (Game object ID={self.game_object_id}) (Current"
            f" texture={self.texture})>"
        )


class Bullet(HadesSprite):
    """Represents a bullet sprite object in the game."""

    def __init__(self: Bullet, registry: Registry, game_object_id: int) -> None:
        """Initialise the object.

        Args:
            registry: The registry that manages the game objects, components, and
            systems.
            game_object_id: The game object's ID.
        """
        super().__init__(
            registry,
            (game_object_id, GameObjectType.Bullet),
            Vec2d(0, 0),
            [make_soft_square_texture(64, color.RED)],
        )

    def __repr__(self: Bullet) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<Bullet (Game object ID={self.game_object_id}) (Current"
            f" texture={self.texture})>"
        )


class AnimatedSprite(HadesSprite):
    """Represents an animated sprite object in the game.

    Attributes:
        sprite_textures: The sprite's textures.
    """

    __slots__ = ("sprite_textures",)

    def __init__(
        self: AnimatedSprite,
        registry: Registry,
        game_object: tuple[int, GameObjectType],
        position: Vec2d,
        textures: list[Texture],
    ) -> None:
        """Initialise the object.

        Args:
            registry: The registry that manages the game objects, components, and
            systems.
            game_object: The game object's ID and type.
            position: The position of the sprite object in the grid.
            textures: The sprite's texture paths.
        """
        super().__init__(registry, game_object, position, textures)
        self.sprite_textures: list[tuple[Texture, Texture]] = [
            (texture, texture.flip_left_right()) for texture in textures
        ]

    def __repr__(self: AnimatedSprite) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<AnimatedSprite (Game object ID={self.game_object_id}) (Current"
            f" texture={self.texture})>"
        )
