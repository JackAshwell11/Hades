"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final, NamedTuple

# Custom
from hades.constants import GameObjectType
from hades_extensions.game_objects import SteeringBehaviours, SteeringMovementState
from hades_extensions.game_objects.components import (
    EffectApplier,
    Footprints,
    Inventory,
    KeyboardMovement,
    MovementForce,
    SteeringMovement,
)

if TYPE_CHECKING:
    from typing import ClassVar

    from hades_extensions.game_objects import ComponentBase

__all__ = ("ENEMY", "FLOOR", "GameObjectConstructor", "PLAYER", "POTION", "WALL")


class GameObjectConstructor(NamedTuple):
    """Represents a constructor for a game object.

    Args:
        game_object_type: The game object's type.
        name: The game object's name.
        textures: The game object's texture paths.
        components: The game object's components.
        blocking: Whether the game object blocks sprite movement or not.
        kinematic: Whether the game object should have a kinematic object or not.
    """

    game_object_type: GameObjectType
    name: str
    textures: list[str]
    components: ClassVar[list[ComponentBase]] = []
    blocking: bool = False
    kinematic: bool = False


# Static tiles
WALL: Final[GameObjectConstructor] = GameObjectConstructor(
    GameObjectType.WALL,
    "Wall",
    ["wall.png"],
    blocking=True,
)
FLOOR: Final[GameObjectConstructor] = GameObjectConstructor(
    GameObjectType.FLOOR,
    "Floor",
    ["floor.png"],
)

# Entities
PLAYER: Final[GameObjectConstructor] = GameObjectConstructor(
    GameObjectType.PLAYER,
    "Player",
    ["player_idle.png"],
    [
        Inventory(6, 5),
        MovementForce(5000, 5),
        KeyboardMovement(),
        Footprints(),
    ],
    kinematic=True,
)
ENEMY: Final[GameObjectConstructor] = GameObjectConstructor(
    GameObjectType.ENEMY,
    "Enemy",
    ["enemy_idle.png"],
    [
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

# Items
POTION: Final[GameObjectConstructor] = GameObjectConstructor(
    GameObjectType.POTION,
    "Health Potion",
    ["health_potion.png"],
    [EffectApplier({}, {})],
)
