"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING, Final, NamedTuple

# Custom
from hades_extensions.game_objects import SteeringBehaviours, SteeringMovementState
from hades_extensions.game_objects.components import (
    Footprints,
    Inventory,
    KeyboardMovement,
    MovementForce,
    SteeringMovement,
)

if TYPE_CHECKING:
    from typing import ClassVar

    from hades_extensions.game_objects import ComponentBase

__all__ = ("GameObjectConstructorManager", "GameObjectType", "COLLECTIBLE_TYPES")


class GameObjectType(Enum):
    """Stores the different types of game objects that can exist in the game."""

    ENEMY = auto()
    FLOOR = auto()
    PLAYER = auto()
    POTION = auto()
    WALL = auto()


class GameObjectConstructor(NamedTuple):
    """Represents a constructor for a game object.

    Args:
        name: The game object's name.
        textures: The game object's texture paths.
        components: The game object's components.
        blocking: Whether the game object blocks sprite movement or not.
        kinematic: Whether the game object should have a kinematic object or not.
    """

    name: str
    textures: list[str]
    components: ClassVar[list[ComponentBase]] = []
    blocking: bool = False
    kinematic: bool = False


class GameObjectConstructorManager:
    """Holds the various templates for game objects and sprites."""

    # Class variables
    _constructors: ClassVar[dict[GameObjectType, GameObjectConstructor]] = {}

    @classmethod
    def add_constructor(
        cls: GameObjectConstructorManager,
        game_object_type: GameObjectType,
        constructor: GameObjectConstructor,
    ) -> None:
        """Add a constructor to the manager.

        Args:
            game_object_type: The type of constructor.
            constructor: The constructor to add.
        """
        cls._constructors[game_object_type] = constructor

    @classmethod
    def get_constructor(
        cls: GameObjectConstructorManager,
        game_object_type: GameObjectType,
    ) -> GameObjectConstructor:
        """Get a constructor from the manager.

        Args:
            game_object_type: The type of game object.

        Returns:
            The constructor.
        """
        return cls._constructors[game_object_type]


# Add the static tiles
GameObjectConstructorManager.add_constructor(
    GameObjectType.WALL,
    GameObjectConstructor(
        "Wall",
        ["wall.png"],
        blocking=True,
    ),
)
GameObjectConstructorManager.add_constructor(
    GameObjectType.FLOOR,
    GameObjectConstructor("Floor", ["floor.png"]),
)

# Add the entities
GameObjectConstructorManager.add_constructor(
    GameObjectType.PLAYER,
    GameObjectConstructor(
        "Player",
        ["player_idle.png"],
        [
            Inventory(6, 5),
            MovementForce(5000, 5),
            KeyboardMovement(),
            Footprints(),
        ],
        kinematic=True,
    ),
)
GameObjectConstructorManager.add_constructor(
    GameObjectType.ENEMY,
    GameObjectConstructor(
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
    ),
)

# Add the items
GameObjectConstructorManager.add_constructor(
    GameObjectType.POTION,
    GameObjectConstructor("Health Potion", ["health_potion.png"]),
)

# Define some collections for game object types
COLLECTIBLE_TYPES: Final[set[GameObjectType]] = {
    GameObjectType.POTION,
}
