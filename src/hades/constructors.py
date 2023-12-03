"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple

# Custom
from hades.constants import GameObjectType
from hades.textures import TextureType
from hades_extensions.game_objects import SteeringBehaviours, SteeringMovementState
from hades_extensions.game_objects.components import (
    Footprints,
    Inventory,
    KeyboardMovement,
    MovementForce,
    SteeringMovement,
)

if TYPE_CHECKING:
    from arcade import Texture

    from hades_extensions.game_objects import ComponentBase

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
        components: A list of components that are part of this game object.
        blocking: Whether the game object blocks sprite movement or not.
        kinematic: Whether the game object should have a kinematic object or not.
    """

    game_object_type: GameObjectType
    game_object_textures: GameObjectTextures
    components: ClassVar[list[ComponentBase]] = []
    blocking: bool = False
    kinematic: bool = False


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
    components=[
        Inventory(6, 5),
        MovementForce(5000, 5),
        KeyboardMovement(),
        Footprints(),
    ],
    kinematic=True,
)

# Enemy characters
ENEMY: Final = GameObjectConstructor(
    GameObjectType.ENEMY,
    GameObjectTextures(TextureType.ENEMY_IDLE.value[0]),
    components=[
        MovementForce(1000, 5),
        SteeringMovement(
            {
                SteeringMovementState.Default: [
                    SteeringBehaviours.ObstacleAvoidance,
                    SteeringBehaviours.Wander,
                ],
                SteeringMovementState.Footprint: [SteeringBehaviours.FollowPath],
                SteeringMovementState.Target: [SteeringBehaviours.Pursue],
            },
        ),
    ],
    kinematic=True,
)

# Potion tiles
POTION: Final = GameObjectConstructor(
    GameObjectType.POTION,
    GameObjectTextures(TextureType.HEALTH_POTION.value),
)

# TODO: This file needs rewriting
