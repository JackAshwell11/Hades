"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, TypedDict

# Pip
from arcade import load_texture

# Custom
from hades.constants import GameObjectType
from hades_extensions.game_objects import (
    SPRITE_SCALE,
    AttackAlgorithm,
    SteeringBehaviours,
    SteeringMovementState,
    Vec2d,
)
from hades_extensions.game_objects.components import (
    Armour,
    Attacks,
    EffectApplier,
    Footprints,
    Health,
    Inventory,
    KeyboardMovement,
    KinematicComponent,
    MovementForce,
    SteeringMovement,
)

if TYPE_CHECKING:
    from typing import ClassVar

    from arcade import Texture

    from hades_extensions.game_objects import ComponentBase

__all__ = ("ENEMY", "FLOOR", "PLAYER", "POTION", "WALL", "GameObjectConstructor")

# Create the texture path
texture_path = Path(__file__).resolve().parent / "resources" / "textures"


class GameObjectConstructor(NamedTuple):
    """Represents a constructor for a game object.

    Args:
        game_object_type: The game object's type.
        name: The game object's name.
        textures: The game object's texture.
        components: The game object's components.
        blocking: Whether the game object blocks sprite movement or not.
    """

    game_object_type: GameObjectType
    name: str
    textures: list[Texture]
    components: ClassVar[list[ComponentBase]]
    blocking: bool


class GameObjectConstructorOptions(TypedDict, total=False):
    """Represents the options when creating a game object constructor.

    Args:
        kinematic: Whether the game object should have a kinematic component or not.
        blocking: Whether the game object blocks sprite movement or not.
    """

    kinematic: bool
    blocking: bool


def create_constructor(
    game_object_type: GameObjectType,
    name: str,
    texture_paths: list[str],
    components: list[ComponentBase] | None = None,
    options: GameObjectConstructorOptions | None = None,
) -> GameObjectConstructor:
    """Creates a constructor for a game object.

    Args:
        game_object_type: The game object's type.
        name: The game object's name.
        texture_paths: The game object's texture paths.
        components: The game object's components.
        options: The game object's options.
    """
    # Set the default values for the parameters
    if components is None:
        components = []
    if options is None:
        options = {}

    # Load the textures and create the constructor
    textures = [load_texture(texture_path.joinpath(path)) for path in texture_paths]
    constructor = GameObjectConstructor(
        game_object_type,
        name,
        textures,
        components,
        options.get("blocking", False),
    )

    # Add the kinematic component if needed using the first texture's hit box
    # points
    if options.get("kinematic", False):
        constructor.components.append(
            KinematicComponent(
                [
                    Vec2d(*hit_box_point) * SPRITE_SCALE
                    for hit_box_point in textures[0].hit_box_points
                ],
            ),
        )
    return constructor


# Static tiles
WALL: Final[GameObjectConstructor] = create_constructor(
    GameObjectType.WALL,
    "Wall",
    ["wall.png"],
    options={
        "blocking": True,
    },
)
FLOOR: Final[GameObjectConstructor] = create_constructor(
    GameObjectType.FLOOR,
    "Floor",
    ["floor.png"],
)


# Entities
PLAYER: Final[GameObjectConstructor] = create_constructor(
    GameObjectType.PLAYER,
    "Player",
    ["player_idle.png"],
    [
        Health(200, 5),
        Armour(100, 5),
        Inventory(6, 5),
        Attacks(
            [
                AttackAlgorithm.Ranged,
                AttackAlgorithm.Melee,
                AttackAlgorithm.AreaOfEffect,
            ],
        ),
        MovementForce(5000, 5),
        KeyboardMovement(),
        Footprints(),
    ],
    {
        "kinematic": True,
    },
)
ENEMY: Final[GameObjectConstructor] = create_constructor(
    GameObjectType.ENEMY,
    "Enemy",
    ["enemy_idle.png"],
    [
        Health(100, 5),
        Armour(50, 5),
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
    {
        "kinematic": True,
    },
)

# Items
POTION: Final[GameObjectConstructor] = create_constructor(
    GameObjectType.POTION,
    "Health Potion",
    ["health_potion.png"],
    [EffectApplier({}, {})],
)

# TODO: Change this so the tests don't have to initialise the textures
