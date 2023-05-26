"""Manages the operations related to the sprite object."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, NamedTuple, cast

# Pip
from arcade import Sprite

# Custom
from hades.constants import SPRITE_SCALE, GameObjectType
from hades.game_objects.attributes import GameObjectAttributeBase
from hades.game_objects.base import ComponentType
from hades.game_objects.movements import MovementManager
from hades.textures import grid_pos_to_pixel

if TYPE_CHECKING:
    from hades.game_objects.constructors import GameObjectTextures
    from hades.game_objects.system import ECS
    from hades.physics import PhysicsEngine

__all__ = (
    "GameObject",
    "HadesSprite",
)


class GameObject(NamedTuple):
    """Stores the details about the ECS game object for a sprite object.

    Args:
        game_object_id: The game object's ID.
        game_object_type: The type of the game object.
        system: The entity component system which manages the game objects.
    """

    game_object_id: int
    game_object_type: GameObjectType
    system: ECS


class HadesSprite(Sprite):
    """Represents a sprite object in the game."""

    def __init__(
        self: HadesSprite,
        game_object: GameObject,
        position: tuple[int, int],
        game_object_textures: GameObjectTextures,
    ) -> None:
        """Initialise the object.

        Args:
            game_object: The details about the ECS game object.
            position: The position of the sprite object in the grid.
            game_object_textures: The collection of textures which relate to this game
            object.
        """
        super().__init__(scale=SPRITE_SCALE)
        self.game_object: GameObject = game_object
        self.position = grid_pos_to_pixel(*position)
        self.game_object_textures: GameObjectTextures = game_object_textures

        # Initialise the default sprite
        self.texture = self.game_object_textures.default_texture

    @property
    def game_object_id(self: HadesSprite) -> int:
        """Get the sprite object's game object ID.

        Returns:
            The sprite object's game object ID.
        """
        return self.game_object.game_object_id

    @property
    def game_object_type(self: HadesSprite) -> GameObjectType:
        """Get the sprite object's game object type.

        Returns:
            The sprite object's game object type.
        """
        return self.game_object.game_object_type

    @property
    def system(self: HadesSprite) -> ECS:
        """Get a reference to the entity component system.

        Returns:
            The reference to the entity component system..
        """
        return self.game_object.system

    @property
    def physics(self: HadesSprite) -> PhysicsEngine:
        """Get the game object's physics engine.

        Returns:
            The game object's physics engine
        """
        return self.physics_engines[0]

    def on_update(self: HadesSprite, delta_time: float = 1 / 60) -> None:
        """Handle an on_update event for the game object.

        Args:
            delta_time: The time interval since the last time the event was triggered.
        """
        # Update the game object's attributes
        for component in self.system.get_game_object_attributes_for_game_object(
            self.game_object_id,
        ):
            cast(GameObjectAttributeBase, component).on_update(delta_time)

        # Calculate the game object's new movement force and apply it
        if movement_manager := cast(
            MovementManager,
            self.system.get_component_for_game_object(
                self.game_object_id,
                ComponentType.MOVEMENT_MANAGER,
            ),
        ):
            self.physics.apply_impulse(
                self,
                movement_manager.get_force(),
            )

    def __repr__(self: HadesSprite) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<HadesSprite (Game object ID={self.game_object_id}) (Game object"
            f" type={self.game_object_type}) (Current texture={self.texture})>"
        )


# TODO: Still need to implement:
#  Armour regeneration (could use idea similar to attack manager) - player and enemy
#  Attack - enemy


# if (
#         self.time_since_armour_regen  # noqa: ERA001
#         >= self.system.get_component_for_game_object(
#     self.game_object_id,  # noqa: ERA001
#     ComponentType.ARMOUR_REGEN_COOLDOWN,  # noqa: ERA001
# ).value
# ):
#     self.system.get_component_for_game_object(
#         self.game_object_id,  # noqa: ERA001
#         ComponentType.ARMOUR,  # noqa: ERA001
#     ).value += ARMOUR_REGEN_AMOUNT
#     self.time_since_armour_regen = 0  # noqa: ERA001
#     self.time_since_armour_regen += delta_time  # noqa: ERA001

# TODO: Maybe add attribute to gameobjectcomponent which says which components should be
#  initialised if that component is provided
