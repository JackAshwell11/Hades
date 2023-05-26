"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, NamedTuple

# Custom
from hades.constants import GameObjectType
from hades.game_objects.components import Inventory
from hades.game_objects.movements import KeyboardMovement, SteeringMovement
from hades.textures import TextureType

if TYPE_CHECKING:
    from arcade import Texture

    from hades.game_objects.base import ComponentData, GameObjectComponent

__all__ = (
    "ENEMY",
    "FLOOR",
    "GameObjectConstructor",
    "GameObjectTextures",
    "PLAYER",
    "POTION",
    "WALL",
)


class GameObjectTextures(NamedTuple):
    """Stores the different textures that a game object can have."""

    default_texture: Texture


class GameObjectConstructor(NamedTuple):
    """Represents a constructor for a game object.

    Args:
        game_object_type: The type of this game object.
        game_object_textures: The collection of textures which relate to this game
        object.
        components: A list of component types that are part of this game object.
        component_data: The data for the components.
        blocking: Whether the game object blocks sprite movement or not.
    """

    game_object_type: GameObjectType
    game_object_textures: GameObjectTextures
    components: list[type[GameObjectComponent]] = []
    component_data: ComponentData = {}
    blocking: bool = False


# Static tiles
WALL = GameObjectConstructor(
    GameObjectType.WALL,
    GameObjectTextures(TextureType.WALL.value),
    blocking=True,
)
FLOOR = GameObjectConstructor(
    GameObjectType.FLOOR,
    GameObjectTextures(TextureType.FLOOR.value),
)

# Player characters
PLAYER = GameObjectConstructor(
    GameObjectType.PLAYER,
    GameObjectTextures(TextureType.PLAYER_IDLE.value[0]),
    components=[Inventory, KeyboardMovement],
    component_data={"inventory_size": (6, 5)},
)

# Enemy characters
ENEMY = GameObjectConstructor(
    GameObjectType.ENEMY,
    GameObjectTextures(TextureType.ENEMY_IDLE.value[0]),
    components=[SteeringMovement],
)

# Potion tiles
POTION = GameObjectConstructor(
    GameObjectType.POTION,
    GameObjectTextures(TextureType.HEALTH_POTION.value),
)
