"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
import json
from typing import TYPE_CHECKING, Final, NamedTuple

# Pip
from arcade import load_texture

# Custom
from hades_extensions.game_objects import (
    SPRITE_SCALE,
    AttackAlgorithm,
    GameObjectType,
    SteeringBehaviours,
    SteeringMovementState,
    Vec2d,
)
from hades_extensions.game_objects.components import (
    Armour,
    Attack,
    EffectApplier,
    Footprints,
    Health,
    Inventory,
    KeyboardMovement,
    KinematicComponent,
    MovementForce,
    StatusEffect,
    SteeringMovement,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ClassVar

    from hades_extensions.game_objects import ComponentBase

__all__ = (
    "BULLET",
    "ENEMY",
    "FLOOR",
    "PLAYER",
    "POTION",
    "WALL",
    "GameObjectConstructor",
    "create_constructor",
)


class GameObjectConstructor(NamedTuple):
    """Represents a constructor for a game object.

    Args:
        name: The game object's name.
        description: The game object's description.
        game_object_type: The game object's type.
        texture_paths: The paths to the game object's textures.
        components: The game object's components.
    """

    name: str
    description: str
    game_object_type: GameObjectType
    texture_paths: list[str]
    components: ClassVar[list[ComponentBase]]


def convert_attack_args(attack_args: dict[str, list[str]]) -> Attack:
    """Convert the attack arguments to the correct format.

    Args:
        attack_args: A list of attack algorithms.

    Raises:
        KeyError: If an attack algorithm is not found.

    Returns:
        The constructed attack component.
    """
    # TODO: Attack should be modified to accept multiple categories (e.g. ranged,
    #  close, special), so `Default` will change
    return Attack(
        [
            AttackAlgorithm.__members__[attack_algorithm]
            for attack_algorithm in attack_args.get("Default", [])
        ],
    )


def convert_steering_movement_args(
    component_args: dict[str, list[str]],
) -> SteeringMovement:
    """Convert the steering movement arguments to the correct format.

    Args:
        component_args: A dictionary of steering movement states and their behaviours.

    Raises:
        KeyError: If a steering movement state or behaviour is not found.

    Returns:
        The constructed steering movement component.
    """
    return SteeringMovement(
        {
            SteeringMovementState.__members__[movement_state]: [
                SteeringBehaviours.__members__[behaviour]
                for behaviour in component_args[movement_state]
            ]
            for movement_state in component_args
        },
    )


# Mapping of component names to classes
COMPONENT_MAPPING: Final[dict[str, type[ComponentBase]]] = {
    "Health": Health,
    "Armour": Armour,
    "Inventory": Inventory,
    "MovementForce": MovementForce,
    "KeyboardMovement": KeyboardMovement,
    "Footprints": Footprints,
    "EffectApplier": EffectApplier,
    "StatusEffect": StatusEffect,
}


# Mapping of component names to their respective conversion classes
CONVERSION_MAPPING: Final[
    dict[str, Callable[[dict[str, list[str]]], ComponentBase]]
] = {
    "Attack": convert_attack_args,
    "SteeringMovement": convert_steering_movement_args,
}


def create_constructor(game_object_json: str) -> GameObjectConstructor:
    """Create a constructor for a game object.

    Args:
        game_object_json: A JSON string that templates a game object.

    Raises:
        ValueError: If the game object type is invalid.
        KeyError: If the game object type or texture paths are not provided or the
        provided component data is invalid.

    Returns:
        The constructed game object constructor.
    """
    # Check if the game object type is valid
    game_object_data = json.loads(game_object_json)
    game_object_type = GameObjectType.__members__.get(
        game_object_data["game_object_type"],
    )
    if game_object_type is None:
        exception = "Invalid game object type"
        raise ValueError(exception)

    # Create the components
    components_dict = game_object_data.get("components", {})
    components = [
        (
            CONVERSION_MAPPING[component_name](component_args)
            if component_name in CONVERSION_MAPPING
            else COMPONENT_MAPPING[component_name](*component_args)
        )
        for component_name, component_args in components_dict.items()
    ]

    # Get the texture paths and add the kinematic component if needed using either the
    # first texture's hit box points or as a static component
    texture_paths = game_object_data["texture_paths"]
    if game_object_data.get("kinematic", False):
        components.append(
            KinematicComponent(
                [
                    Vec2d(*hit_box_point) * SPRITE_SCALE
                    for hit_box_point in load_texture(texture_paths[0]).hit_box_points
                ],
            ),
        )
    elif game_object_data.get("static", False):
        components.append(KinematicComponent(is_static=True))
    return GameObjectConstructor(
        game_object_data["name"],
        game_object_data["description"],
        game_object_type,
        texture_paths,
        components,
    )


# Static tiles
WALL: Final[str] = json.dumps(
    {
        "name": "Wall",
        "description": "A wall that blocks movement.",
        "game_object_type": "Wall",
        "texture_paths": [":resources:wall.png"],
        "static": True,
    },
)

FLOOR: Final[str] = json.dumps(
    {
        "name": "Floor",
        "description": "A floor that allows movement.",
        "game_object_type": "Floor",
        "texture_paths": [":resources:floor.png"],
    },
)

# Entities
PLAYER: Final[str] = json.dumps(
    {
        "name": "Player",
        "description": "The player character.",
        "game_object_type": "Player",
        "texture_paths": [":resources:player_idle.png"],
        "components": {
            "Health": [200, 5],
            "Armour": [100, 5],
            "Inventory": [6, 5],
            "Attack": {
                "Default": ["Ranged", "Melee", "AreaOfEffect"],
            },
            "MovementForce": [5000, 5],
            "KeyboardMovement": [],
            "Footprints": [],
            "StatusEffect": [],
        },
        "kinematic": True,
    },
)

ENEMY: Final[str] = json.dumps(
    {
        "name": "Enemy",
        "description": "An enemy character.",
        "game_object_type": "Enemy",
        "texture_paths": [":resources:enemy_idle.png"],
        "components": {
            "Health": [100, 5],
            "Armour": [50, 5],
            "Attack": {
                "Default": ["Ranged"],
            },
            "MovementForce": [1000, 5],
            "SteeringMovement": {
                "Default": ["ObstacleAvoidance", "Wander"],
                "Footprint": ["FollowPath"],
                "Target": ["Pursue"],
            },
        },
        "kinematic": True,
    },
)

# Items
POTION: Final[str] = json.dumps(
    {
        "name": "Health Potion",
        "description": "A potion that restores health.",
        "game_object_type": "Potion",
        "texture_paths": [":resources:health_potion.png"],
        "components": {
            "EffectApplier": [{}, {}],
        },
    },
)

# Other
BULLET: Final[str] = json.dumps(
    {
        "name": "Bullet",
        "description": "A bullet that damages other game objects.",
        "game_object_type": "Bullet",
        "texture_paths": [":resources:bullet.png"],
        "components": {},
    },
)

# TODO: Add error handling
