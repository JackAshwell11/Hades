"""Manages the operations related to the sprite object."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import BasicSprite, Texture

# Custom
from hades_extensions.ecs import SPRITE_SCALE, GameObjectType, Registry
from hades_extensions.ecs.components import KinematicComponent

if TYPE_CHECKING:
    from hades.constructors import GameObjectConstructor

__all__ = ("AnimatedSprite", "DynamicSprite", "HadesSprite", "make_sprite")


class HadesSprite(BasicSprite):
    """Represents a sprite object in the game."""

    __slots__ = ("constructor", "game_object_id", "registry")

    def __init__(
        self: HadesSprite,
        registry: Registry,
        game_object_id: int,
        position: tuple[float, float],
        constructor: GameObjectConstructor,
    ) -> None:
        """Initialise the object.

        Args:
            registry: The registry that manages the game objects, components, and
                systems.
            game_object_id: The game object's ID.
            position: The sprite's initial position.
            constructor: The game object's constructor.
        """
        super().__init__(
            constructor.textures[0].get_texture(),
            SPRITE_SCALE,
            position[0],
            position[1],
        )
        self.registry: Registry = registry
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


class DynamicSprite(HadesSprite):
    """Represents a dynamic sprite object in the game."""

    def update(self: DynamicSprite, *_: tuple[float]) -> None:
        """Update the sprite object."""
        self.position = self.registry.get_component(
            self.game_object_id,
            KinematicComponent,
        ).get_position()


class AnimatedSprite(DynamicSprite):
    """Represents an animated sprite object in the game.

    Attributes:
        sprite_textures: The sprite's textures.
    """

    __slots__ = ("sprite_textures",)

    def __init__(
        self: AnimatedSprite,
        registry: Registry,
        game_object_id: int,
        position: tuple[float, float],
        constructor: GameObjectConstructor,
    ) -> None:
        """Initialise the object.

        Args:
            registry: The registry that manages the game objects, components, and
                systems.
            game_object_id: The game object's ID.
            position: The sprite's initial position.
            constructor: The game object's constructor.
        """
        super().__init__(registry, game_object_id, position, constructor)
        self.sprite_textures: list[tuple[Texture, Texture]] = [
            (texture.get_texture(), texture.get_texture().flip_left_right())
            for texture in constructor.textures
        ]


def make_sprite(
    registry: Registry,
    game_object_id: int,
    position: tuple[float, float],
    constructor: GameObjectConstructor,
) -> HadesSprite:
    """Create a sprite object.

    Args:
        registry: The registry that manages the game objects, components, and systems.
        game_object_id: The game object's ID.
        position: The sprite's initial position.
        constructor: The game object's constructor.

    Returns:
        The sprite object.
    """
    sprite_class: type[HadesSprite]
    if constructor.game_object_type == GameObjectType.Bullet:
        sprite_class = DynamicSprite
    elif len(constructor.textures) > 1:
        sprite_class = AnimatedSprite
    else:
        sprite_class = HadesSprite
    return sprite_class(registry, game_object_id, position, constructor)
