"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

# Pip
from arcade import Texture, load_texture

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

__all__ = (
    "GameObjectConstructorManager",
    "GameObjectType",
    "PHYSICS_CONSTRUCTORS",
    "STATIC_CONSTRUCTORS",
)

# Create the texture path
texture_path = Path(__file__).resolve().parent / "resources" / "textures"


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
        textures: The game object's textures.
        components: The game object's components.
        kinematic: Whether the game object should have a kinematic object or not.
    """

    name: str
    textures: list[str]
    components: ClassVar[list[ComponentBase]] = []
    kinematic: bool = False


class GameObjectConstructorManager:
    """Holds the various templates for game objects and sprites."""

    _static_constructors: ClassVar[dict[GameObjectType, tuple[Texture, bool]]] = {}
    _dynamic_constructors: ClassVar[dict[GameObjectType, GameObjectConstructor]] = {}

    @classmethod
    def add_static_constructor(
        cls: GameObjectConstructorManager,
        game_object_type: GameObjectType,
        texture: str,
        *,
        blocking: bool = False,
    ) -> None:
        """Add a static constructor to the manager.

        Args:
            game_object_type: The type of constructor.
            texture: The game object's texture.
            blocking: Whether the game object blocks sprite movement or not.
        """
        cls._static_constructors[game_object_type] = (
            load_texture(texture_path.joinpath(texture)),
            blocking,
        )

    @classmethod
    def add_dynamic_constructor(
        cls: GameObjectConstructorManager,
        game_object_type: GameObjectType,
        constructor: GameObjectConstructor,
    ) -> None:
        """Add a dynamic constructor to the manager.

        Args:
            game_object_type: The type of constructor.
            constructor: The constructor to add.
        """
        cls._dynamic_constructors[game_object_type] = constructor

    @classmethod
    def get_static_constructor(
        cls: GameObjectConstructorManager,
        game_object_type: GameObjectType,
    ) -> tuple[Texture, bool]:
        """Get a static constructor from the manager.

        Args:
            game_object_type: The type of game object.

        Returns:
            The static constructor.
        """
        return cls._static_constructors[game_object_type]

    @classmethod
    def get_dynamic_constructor(
        cls: GameObjectConstructorManager,
        game_object_type: GameObjectType,
    ) -> GameObjectConstructor:
        """Get a dynamic constructor from the manager.

        Args:
            game_object_type: The type of game object.

        Returns:
            The dynamic constructor.
        """
        return cls._dynamic_constructors[game_object_type]


# Add the static tiles
GameObjectConstructorManager.add_static_constructor(
    GameObjectType.WALL,
    "wall.png",
    blocking=True,
)
GameObjectConstructorManager.add_static_constructor(GameObjectType.FLOOR, "floor.png")

# Add the entities
GameObjectConstructorManager.add_dynamic_constructor(
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
GameObjectConstructorManager.add_dynamic_constructor(
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
GameObjectConstructorManager.add_dynamic_constructor(
    GameObjectType.POTION,
    GameObjectConstructor("Health Potion", ["health_potion.png"]),
)

# Define some collections for game object types
STATIC_CONSTRUCTORS: Final[set[GameObjectType]] = {
    GameObjectType.FLOOR,
    GameObjectType.WALL,
}
PHYSICS_CONSTRUCTORS: Final[set[GameObjectType]] = {
    GameObjectType.ENEMY,
    GameObjectType.PLAYER,
    GameObjectType.WALL,
}
