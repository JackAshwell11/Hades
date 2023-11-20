"""Manages the operations related to the sprite object."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import Sprite

# Custom
from hades.textures import grid_pos_to_pixel
from hades_extensions.game_objects import SPRITE_SCALE, Vec2d
from hades_extensions.game_objects.systems import (
    KeyboardMovementSystem,
    SteeringMovementSystem,
)

if TYPE_CHECKING:
    from hades.constructors import GameObjectTextures
    from hades.physics import PhysicsEngine
    from hades_extensions.game_objects import Registry

__all__ = ("HadesSprite",)


class HadesSprite(Sprite):
    """Represents a sprite object in the game."""

    def __init__(
        self: HadesSprite,
        game_object: tuple[int, bool],
        registry: Registry,
        position: tuple[int, int],
        game_object_textures: GameObjectTextures,
    ) -> None:
        """Initialise the object.

        Args:
            game_object: The game object's ID and whether it is AI controlled or not.
            registry: The registry which manages the game objects.
            position: The position of the sprite object in the grid.
            game_object_textures: The collection of textures which relate to this game
            object.
        """
        super().__init__(scale=SPRITE_SCALE)
        self.game_object_id, ai_controlled = game_object
        self.registry: Registry = registry
        self.position = grid_pos_to_pixel(*position)
        self.game_object_textures: GameObjectTextures = game_object_textures
        self.in_combat: bool = False

        # Initialise the default sprite
        self.texture = self.game_object_textures.default_texture

        # Get the correct movement system for the game object
        self.target_movement_system: KeyboardMovementSystem | SteeringMovementSystem = (
            self.registry.get_system(SteeringMovementSystem)
            if ai_controlled
            else self.registry.get_system(KeyboardMovementSystem)
        )

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
        force = self.target_movement_system.calculate_force(self.game_object_id)
        self.physics.apply_force(
            self,
            (force.x, force.y),
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
        kinematic_object, body = (
            self.registry.get_kinematic_object(self.game_object_id),
            physics_engine.get_physics_object(self).body,
        )
        if body is None:
            return
        kinematic_object.position = Vec2d(*body.position)
        kinematic_object.velocity = Vec2d(*body.velocity)

    def __repr__(self: HadesSprite) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<HadesSprite (Game object ID={self.game_object_id}) (Current"
            f" texture={self.texture})>"
        )
