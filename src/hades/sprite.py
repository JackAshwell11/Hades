"""Manages the operations related to the sprite object."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

# Pip
from arcade import Sprite

# Custom
from hades.constants import SPRITE_SCALE
from hades.game_objects.base import ComponentType
from hades.game_objects.movements import MovementBase
from hades.textures import grid_pos_to_pixel

if TYPE_CHECKING:
    from hades.game_objects.constructors import GameObjectTextures
    from hades.game_objects.registry import Registry
    from hades.physics import PhysicsEngine

__all__ = ("HadesSprite",)


class HadesSprite(Sprite):
    """Represents a sprite object in the game."""

    def __init__(
        self: HadesSprite,
        game_object_id: int,
        registry: Registry,
        position: tuple[int, int],
        game_object_textures: GameObjectTextures,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object's ID.
            registry: The registry which manages the game objects.
            position: The position of the sprite object in the grid.
            game_object_textures: The collection of textures which relate to this game
            object.
        """
        super().__init__(scale=SPRITE_SCALE)
        self.game_object_id: int = game_object_id
        self.registry: Registry = registry
        self.position = grid_pos_to_pixel(*position)
        self.game_object_textures: GameObjectTextures = game_object_textures
        self.in_combat: bool = False

        # Initialise the default sprite
        self.texture = self.game_object_textures.default_texture

    @property
    def physics(self: HadesSprite) -> PhysicsEngine:
        """Get the game object's physics engine.

        Returns:
            The game object's physics engine
        """
        return self.physics_engines[0]  # type: ignore[misc,no-any-return]

    def on_update(self: HadesSprite, _: float = 1 / 60) -> None:
        """Handle an on_update event for the game object."""
        # Calculate the game object's new movement force and apply it
        self.physics.apply_force(
            self,
            cast(
                MovementBase,
                self.registry.get_component_for_game_object(
                    self.game_object_id,
                    ComponentType.MOVEMENTS,
                ),
            ).calculate_force(),
        )

    def pymunk_moved(
        self: HadesSprite,
        physics_engine: PhysicsEngine,
        *_: float,
    ) -> None:
        """Handle a pymunk_moved event for the game object.

        Args:
            physics_engine: The game object's physics engine.
        """
        physics_object, body = (
            self.registry.get_physics_object_for_game_object(self.game_object_id),
            physics_engine.get_physics_object(self).body,
        )
        if body is None:
            return
        physics_object.position = body.position
        physics_object.velocity = body.velocity

    def __repr__(self: HadesSprite) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<HadesSprite (Game object ID={self.game_object_id}) (Current"
            f" texture={self.texture})>"
        )
