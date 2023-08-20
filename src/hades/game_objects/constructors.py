"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple

# Custom
from hades.constants import GameObjectType
from hades.game_objects.base import (
    ComponentType,
    SteeringBehaviours,
    SteeringMovementState,
)
from hades.game_objects.components import Footprint, Inventory
from hades.game_objects.movements import KeyboardMovement, SteeringMovement
from hades.game_objects.systems import MovementForce
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
        physics: Whether the game object should have a physics object or not.
    """

    game_object_type: GameObjectType
    game_object_textures: GameObjectTextures
    components: ClassVar[list[type[GameObjectComponent]]] = []
    component_data: ClassVar[ComponentData] = {}
    blocking: bool = False
    physics: bool = False


# Static tiles
WALL: Final = GameObjectConstructor(
    GameObjectType.WALL,
    GameObjectTextures(TextureType.WALL.value),
    blocking=True,
)
FLOOR: Final = GameObjectConstructor(
    GameObjectType.FLOOR,
    GameObjectTextures(TextureType.FLOOR.value),
)

# Player characters
PLAYER: Final = GameObjectConstructor(
    GameObjectType.PLAYER,
    GameObjectTextures(TextureType.PLAYER_IDLE.value[0]),
    components=[Inventory, MovementForce, KeyboardMovement, Footprint],
    component_data={
        "attributes": {ComponentType.MOVEMENT_FORCE: (5000, 5)},
        "inventory_size": (6, 5),
    },
    physics=True,
)

# Enemy characters
ENEMY: Final = GameObjectConstructor(
    GameObjectType.ENEMY,
    GameObjectTextures(TextureType.ENEMY_IDLE.value[0]),
    components=[MovementForce, SteeringMovement],
    component_data={
        "attributes": {ComponentType.MOVEMENT_FORCE: (1000, 5)},
        "steering_behaviours": {
            SteeringMovementState.DEFAULT: [
                SteeringBehaviours.OBSTACLE_AVOIDANCE,
                SteeringBehaviours.WANDER,
            ],
            SteeringMovementState.FOOTPRINT: [SteeringBehaviours.FOLLOW_PATH],
            SteeringMovementState.TARGET: [SteeringBehaviours.PURSUIT],
        },
    },
    physics=True,
)

# Potion tiles
POTION: Final = GameObjectConstructor(
    GameObjectType.POTION,
    GameObjectTextures(TextureType.HEALTH_POTION.value),
)
