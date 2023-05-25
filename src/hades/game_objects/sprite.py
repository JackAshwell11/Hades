"""Manages the operations related to the sprite object."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, NamedTuple

# Pip
from arcade import Sprite

# Custom
from hades.constants import SPRITE_SCALE, ComponentType, GameObjectType
from hades.textures import grid_pos_to_pixel

if TYPE_CHECKING:
    from hades.game_objects.constructors import GameObjectTextures
    from hades.game_objects.system import ECS

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

    def on_update(self: HadesSprite, delta_time: float = 1 / 60) -> None:
        """Handle an on_update event for the game object.

        Args:
            delta_time: The time interval since the last time the event was triggered.
        """
        # Update the game object's status effects
        for component in self.system.get_entity_attributes_for_game_object(
            self.game_object_id,
        ):
            component.update_status_effect(delta_time)

        # Calculate the game object's new movement force and apply it
        # TODO: Needs to handle:
        #  Movement - player and enemy
        #  Attack - enemy

        # TODO: Could find way to move enemy attack out of here so game_object_type
        #  isn't needed

        # TODO: Restructure where and how armour regeneration is done

    def deal_damage(self: HadesSprite, damage: int) -> None:
        """Deal damage to the game object.

        Args:
            damage: The damage to deal to the game object.
        """
        # Damage the armour and carry over the extra damage to the health
        health, armour = self.system.get_component_for_game_object(
            self.game_object_id,
            ComponentType.HEALTH,
        ), self.system.get_component_for_game_object(
            self.game_object_id,
            ComponentType.ARMOUR,
        )
        health.value -= max(damage - armour.value, 0)
        armour.value -= damage

    def __repr__(self: HadesSprite) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<HadesSprite (Game object ID={self.game_object_id}) (Game object"
            f" type={self.game_object_type}) (Current texture={self.texture})>"
        )
