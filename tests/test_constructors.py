"""Tests all classes and functions in constructors.py."""

from __future__ import annotations

# Builtin
import json
from pathlib import Path

# Pip
import pytest

# Custom
from hades.constructors import create_constructor
from hades_extensions.game_objects import GameObjectType
from hades_extensions.game_objects.components import Attack, Stat, SteeringMovement


def test_create_constructor_valid_game_object_type() -> None:
    """Test create_constructor() with a valid game object type."""
    constructor = create_constructor(
        json.dumps(
            {
                "name": "Player",
                "description": "The player",
                "game_object_type": "Player",
                "texture_paths": [],
            },
        ),
    )
    assert constructor.game_object_type == GameObjectType.Player


def test_create_constructor_invalid_game_object_type() -> None:
    """Test create_constructor() with an invalid game object type."""
    with pytest.raises(expected_exception=ValueError, match="Invalid game object type"):
        create_constructor(
            json.dumps(
                {
                    "name": "Player",
                    "description": "The player",
                    "game_object_type": "Test",
                },
            ),
        )


@pytest.mark.parametrize(
    ("texture_paths", "expected_length"),
    [
        ([], 0),
        (["floor.png"], 1),
        (["floor.png", "wall.png"], 2),
    ],
)
def test_create_constructor_texture_paths(
    texture_paths: list[str],
    expected_length: int,
) -> None:
    """Test create_constructor() with various texture paths.

    Args:
        texture_paths: The texture paths to test with.
        expected_length: The expected length of the texture paths.
    """
    constructor = create_constructor(
        json.dumps(
            {
                "name": "Player",
                "description": "The player",
                "game_object_type": "Player",
                "texture_paths": texture_paths,
            },
        ),
    )
    assert len(constructor.texture_paths) == expected_length


@pytest.mark.parametrize(
    ("generic_components", "args"),
    [
        ([], []),
        (["Health"], [[200, 5]]),
        (["Health", "Armour"], [[200, 5], [100, 10]]),
    ],
)
def test_create_constructor_generic_components(
    generic_components: list[str],
    args: list[list[int]],
) -> None:
    """Test create_constructor() with generic components.

    Args:
        generic_components: The generic components to test with.
        args: The arguments for each generic component.
    """
    constructor = create_constructor(
        json.dumps(
            {
                "name": "Player",
                "description": "The player",
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": dict(zip(generic_components, args)),
            },
        ),
    )
    assert len(constructor.components) == len(generic_components)
    for index, component in enumerate(constructor.components):
        assert isinstance(component, Stat)
        assert component.get_value() == args[index][0]
        assert component.get_max_level() == args[index][1]


@pytest.mark.parametrize(
    "attack_args",
    [
        ({}),
        ({"Default": []}),
        ({"Default": ["Ranged"]}),
        ({"Default": ["Ranged", "Melee"]}),
        ({"Default": ["Ranged"], "Special": ["Melee"]}),
        ({"Test": []}),
    ],
)
def test_create_constructor_attack(attack_args: dict[str, list[str]]) -> None:
    """Test create_constructor() with various attack components.

    Args:
        attack_args: The arguments for the attack component.
    """
    constructor = create_constructor(
        json.dumps(
            {
                "name": "Player",
                "description": "The player",
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"Attack": attack_args},
            },
        ),
    )
    assert isinstance(constructor.components[0], Attack)


@pytest.mark.parametrize(
    "steering_movement_args",
    [
        ({}),
        ({"Default": []}),
        ({"Default": ["Wander"]}),
        ({"Default": ["ObstacleAvoidance", "Wander"]}),
        ({"Default": ["Wander"], "Target": ["Pursue"]}),
    ],
)
def test_create_constructor_steering_movement(
    steering_movement_args: dict[str, list[str]],
) -> None:
    """Test create_constructor() with various steering movement components.

    Args:
        steering_movement_args: The arguments for the steering movement component.
    """
    constructor = create_constructor(
        json.dumps(
            {
                "name": "Player",
                "description": "The player",
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"SteeringMovement": steering_movement_args},
            },
        ),
    )
    assert isinstance(constructor.components[0], SteeringMovement)


@pytest.mark.parametrize(
    ("kinematic", "static"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_create_constructor_kinematic_static(*, kinematic: bool, static: bool) -> None:
    """Test create_constructor() with various kinematic and static properties.

    Args:
        kinematic: The kinematic property of the game object.
        static: The static property of the game object.
    """
    game_object_data = {
        "name": "Player",
        "description": "The player",
        "game_object_type": "Player",
        "texture_paths": [
            str(
                Path(__file__).resolve().parent.parent
                / "src"
                / "hades"
                / "resources"
                / "textures"
                / "floor.png",
            ),
        ],
        "kinematic": kinematic,
        "static": static,
    }
    constructor = create_constructor(json.dumps(game_object_data))
    assert constructor.components[0]


@pytest.mark.parametrize(
    ("component_args", "expected_exception", "expected_message"),
    [
        ({"Test": []}, KeyError, "Test"),
        ({"Health": []}, TypeError, ".*incompatible constructor arguments.*"),
        ({"Attack": {"Default": ["Test"]}}, KeyError, "Test"),
        ({"SteeringMovement": {"Test": []}}, KeyError, "Test"),
        ({"SteeringMovement": {"Default": ["Test"]}}, KeyError, "Test"),
    ],
)
def test_create_constructor_invalid_component_args(
    component_args: dict[str, list[int] | dict[str, list[str]]],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """Test create_constructor() with invalid component arguments.

    Args:
        component_args: The arguments for the component.
        expected_exception: The expected exception.
        expected_message: The expected exception message.
    """
    with pytest.raises(expected_exception, match=expected_message):
        create_constructor(
            json.dumps(
                {
                    "name": "Player",
                    "description": "The player",
                    "game_object_type": "Player",
                    "texture_paths": ["floor.png"],
                    "components": component_args,
                },
            ),
        )


@pytest.mark.parametrize(
    ("keys_to_remove", "expected_exception", "expected_message"),
    [
        (["game_object_type"], KeyError, "game_object_type"),
        (["name"], KeyError, "name"),
        (["description"], KeyError, "description"),
        (["texture_paths"], KeyError, "texture_paths"),
        (
            ["game_object_type", "name", "description", "texture_paths"],
            KeyError,
            "game_object_type",
        ),
    ],
)
def test_create_constructor_missing_keys(
    keys_to_remove: list[str],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """Test create_constructor() with missing keys.

    Args:
        keys_to_remove: The keys to remove from the data.
        expected_exception: The expected exception.
        expected_message: The expected exception message.
    """
    data = {
        "name": "Player",
        "description": "The player",
        "game_object_type": "Player",
        "texture_paths": [],
    }
    with pytest.raises(expected_exception, match=expected_message):
        create_constructor(
            json.dumps(
                {
                    key: value
                    for key, value in data.items()
                    if key not in keys_to_remove
                },
            ),
        )
