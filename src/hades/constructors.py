"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
import json
from pathlib import Path
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
    from collections.abc import Callable
    from typing import ClassVar

    from arcade import Texture

    from hades_extensions.game_objects import ComponentBase

__all__ = (
    "ENEMY",
    "FLOOR",
    "PLAYER",
    "POTION",
    "WALL",
    "GameObjectConstructor",
    "create_constructor",
)

# Create the texture path
texture_path = Path(__file__).resolve().parent / "resources" / "textures"


class GameObjectConstructor(NamedTuple):
    """Represents a constructor for a game object.

    Args:
        game_object_type: The game object's type.
        textures: The game object's texture.
        components: The game object's components.
    """

    game_object_type: GameObjectType
    textures: list[Texture]
    components: ClassVar[list[ComponentBase]]


def convert_attacks_args(attack_args: list[str]) -> Attacks:
    """Convert the attack arguments to the correct format.

    Args:
        attack_args: A list of attack algorithms.

    Raises:
        KeyError: If an attack algorithm is not found.

    Returns:
        The constructed attacks component.
    """
    return Attacks(
        [
            AttackAlgorithm.__members__.get(attack_algorithm)
            for attack_algorithm in attack_args
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
            SteeringMovementState.__members__.get(movement_state): [
                SteeringBehaviours.__members__.get(behaviour)
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
}


# Mapping of component names to their respective conversion classes
CONVERSION_MAPPING: Final[
    dict[str, Callable[[list[str] | dict[str, list[str]]], ComponentBase]],
] = {
    "Attacks": convert_attacks_args,
    "SteeringMovement": convert_steering_movement_args,
}


def create_constructor(game_object_json: str) -> GameObjectConstructor:
    """Create a constructor for a game object.

    Args:
        game_object_json: A JSON string that templates a game object.

    Raises:
        ValueError: If the game object type is invalid.
        KeyError: If the game object type or texture paths are not provided.
        TypeError: If the provided component data is invalid.

    Returns:
        The constructed game object constructor.
    """
    # Parse the JSON string and get the values
    game_object_data = json.loads(game_object_json)
    game_object_type = GameObjectType.__members__.get(
        game_object_data["game_object_type"],
    )
    if game_object_type is None:
        exception = "Invalid game object type"
        raise ValueError(exception)
    texture_paths = game_object_data["texture_paths"]
    components_dict = game_object_data.get("components", {})
    kinematic = game_object_data.get("kinematic", False)
    static = game_object_data.get("static", False)

    # Load the textures and create the components
    textures = [load_texture(texture_path.joinpath(path)) for path in texture_paths]
    components = [
        (
            CONVERSION_MAPPING[component_name](component_args)
            if component_name in CONVERSION_MAPPING
            else COMPONENT_MAPPING[component_name](*component_args)
        )
        for component_name, component_args in components_dict.items()
    ]

    # Add the kinematic component if needed using either the first texture's hit box
    # points or a static component
    if kinematic:
        components.append(
            KinematicComponent(
                [
                    Vec2d(*hit_box_point) * SPRITE_SCALE
                    for hit_box_point in textures[0].hit_box_points
                ],
            ),
        )
    elif static:
        components.append(KinematicComponent(is_static=True))
    return GameObjectConstructor(game_object_type, textures, components)


# Static tiles
WALL: Final[str] = json.dumps(
    {
        "game_object_type": "Wall",
        "texture_paths": ["wall.png"],
        "static": True,
    },
)

FLOOR: Final[str] = json.dumps(
    {
        "game_object_type": "Floor",
        "texture_paths": ["floor.png"],
    },
)

# Entities
PLAYER: Final[str] = json.dumps(
    {
        "game_object_type": "Player",
        "texture_paths": ["player_idle.png"],
        "components": {
            "Health": [200, 5],
            "Armour": [100, 5],
            "Inventory": [6, 5],
            "Attacks": ["Ranged", "Melee", "AreaOfEffect"],
            "MovementForce": [5000, 5],
            "KeyboardMovement": [],
            "Footprints": [],
            "SteeringMovement": {
                "Default": ["ObstacleAvoidance", "Wander"],
                "Footprint": ["FollowPath"],
                "Target": ["Pursue"],
            },
        },
        "kinematic": True,
    },
)

ENEMY: Final[str] = json.dumps(
    {
        "game_object_type": "Enemy",
        "texture_paths": ["enemy_idle.png"],
        "components": {
            "Health": [100, 5],
            "Armour": [50, 5],
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
        "game_object_type": "Potion",
        "name": "Health Potion",
        "texture_paths": ["health_potion.png"],
        "components": {
            "EffectApplier": [{}, {}],
        },
    },
)

# TODO: Add error handling
