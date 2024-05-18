"""Tests all classes and functions in constructors.py."""

from __future__ import annotations

# Builtin
import json
from typing import cast

# Pip
import pytest

# Custom
from hades.constructors import create_constructor
from hades_extensions.game_objects import GameObjectType
from hades_extensions.game_objects.components import (
    Armour,
    Attacks,
    Health,
    SteeringMovement,
)


def test_create_constructor_empty() -> None:
    """Test create_constructor() with no data."""
    with pytest.raises(expected_exception=KeyError, match="game_object_type"):
        create_constructor(json.dumps({}))


def test_create_constructor_invalid_game_object_type() -> None:
    """Test create_constructor() with an invalid game object type."""
    with pytest.raises(expected_exception=ValueError, match="Invalid game object type"):
        create_constructor(json.dumps({"game_object_type": "Test"}))


def test_create_constructor_valid_game_object_type() -> None:
    """Test create_constructor() with a valid game object type."""
    constructor = create_constructor(
        json.dumps({"game_object_type": "Player", "texture_paths": []}),
    )
    assert constructor.game_object_type == GameObjectType.Player


def test_create_constructor_no_textures() -> None:
    """Test create_constructor() with no textures."""
    with pytest.raises(expected_exception=KeyError, match="texture_paths"):
        create_constructor(json.dumps({"game_object_type": "Player"}))


def test_create_constructor_empty_textures() -> None:
    """Test create_constructor() with empty textures."""
    constructor = create_constructor(
        json.dumps({"game_object_type": "Player", "texture_paths": []}),
    )
    assert constructor.textures == []


def test_create_constructor_single_texture() -> None:
    """Test create_constructor() with a single texture."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
            },
        ),
    )
    assert len(constructor.textures) == 1


def test_create_constructor_multiple_textures() -> None:
    """Test create_constructor() with multiple textures."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png", "wall.png"],
            },
        ),
    )
    assert len(constructor.textures) == 2


def test_create_constructor_no_components() -> None:
    """Test create_constructor() with no components."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
            },
        ),
    )
    assert constructor.components == []


def test_create_constructor_single_generic_component_no_args() -> None:
    """Test create_constructor() with a single component."""
    with pytest.raises(
        expected_exception=TypeError,
        match=".*incompatible constructor arguments.*",
    ):
        create_constructor(
            json.dumps(
                {
                    "game_object_type": "Player",
                    "texture_paths": ["floor.png"],
                    "components": {"Health": []},
                },
            ),
        )


def test_create_constructor_single_generic_component_args() -> None:
    """Test create_constructor() with a single component and args."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"Health": [200, 5]},
            },
        ),
    )
    health_component = cast(Health, constructor.components[0])
    health_component.get_max_level()
    assert health_component.get_value() == 200
    assert health_component.get_max_level() == 5


def test_create_constructor_multiple_generic_components() -> None:
    """Test create_constructor() with multiple components."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {
                    "Health": [200, 5],
                    "Armour": [100, 10],
                },
            },
        ),
    )
    health_component = cast(Health, constructor.components[0])
    armour_component = cast(Armour, constructor.components[1])
    assert health_component.get_value() == 200
    assert health_component.get_max_level() == 5
    assert armour_component.get_value() == 100


def test_create_constructor_attacks_empty() -> None:
    """Test create_constructor() with an attacks component that is empty."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"Attacks": {}},
            },
        ),
    )
    assert isinstance(constructor.components[0], Attacks)


def test_create_constructor_attacks_single_state_empty() -> None:
    """Test create_constructor() with an attacks component with a single attack."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"Attacks": {"Default": []}},
            },
        ),
    )
    assert isinstance(constructor.components[0], Attacks)


def test_create_constructor_attacks_single_state_single() -> None:
    """Test create_constructor() with an attacks component with a single attack."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"Attacks": {"Default": ["Ranged"]}},
            },
        ),
    )
    assert isinstance(constructor.components[0], Attacks)


def test_create_constructor_attacks_single_state_multiple() -> None:
    """Test create_constructor() with an attacks component with a single attack."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"Attacks": {"Default": ["Ranged", "Melee"]}},
            },
        ),
    )
    assert isinstance(constructor.components[0], Attacks)


def test_create_constructor_attacks_multiple() -> None:
    """Test create_constructor() with an attacks component with multiple attacks."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {
                    "Attacks": {"Default": ["Ranged"], "Special": ["Melee"]},
                },
            },
        ),
    )
    assert isinstance(constructor.components[0], Attacks)


def test_create_constructor_attacks_extra_state() -> None:
    """Test create_constructor() with an attacks component with an extra state."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"Attacks": {"Test": []}},
            },
        ),
    )

    assert isinstance(constructor.components[0], Attacks)


def test_create_constructor_attacks_invalid_attack() -> None:
    """Test create_constructor() with an attacks component with invalid data."""
    with pytest.raises(
        expected_exception=KeyError,
        match="Test",
    ):
        create_constructor(
            json.dumps(
                {
                    "game_object_type": "Player",
                    "texture_paths": ["floor.png"],
                    "components": {"Attacks": {"Default": ["Test"]}},
                },
            ),
        )


def test_create_constructor_steering_movement_empty() -> None:
    """Test create_constructor() with a steering movement component that is empty."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"SteeringMovement": {}},
            },
        ),
    )
    assert isinstance(constructor.components[0], SteeringMovement)


def test_create_constructor_steering_movement_single_state_empty() -> None:
    """Test create_constructor() with a single state steering movement component."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"SteeringMovement": {"Default": []}},
            },
        ),
    )
    assert isinstance(constructor.components[0], SteeringMovement)


def test_create_constructor_steering_movement_single_state_single() -> None:
    """Test create_constructor() with a single state steering movement component."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {"SteeringMovement": {"Default": ["Wander"]}},
            },
        ),
    )
    assert isinstance(constructor.components[0], SteeringMovement)


def test_create_constructor_steering_movement_single_state_multiple() -> None:
    """Test create_constructor() with a single state steering movement component."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {
                    "SteeringMovement": {"Default": ["ObstacleAvoidance", "Wander"]},
                },
            },
        ),
    )
    assert isinstance(constructor.components[0], SteeringMovement)


def test_create_constructor_steering_movement_multiple_states() -> None:
    """Test create_constructor() with a multiple state steering movement component."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "components": {
                    "SteeringMovement": {
                        "Default": ["Wander"],
                        "Target": ["Pursue"],
                    },
                },
            },
        ),
    )
    assert isinstance(constructor.components[0], SteeringMovement)


def test_create_constructor_steering_movement_invalid_state() -> None:
    """Test create_constructor() with an invalid steering movement state."""
    with pytest.raises(
        expected_exception=KeyError,
        match="Test",
    ):
        create_constructor(
            json.dumps(
                {
                    "game_object_type": "Player",
                    "texture_paths": ["floor.png"],
                    "components": {
                        "SteeringMovement": {
                            "Test": [],
                        },
                    },
                },
            ),
        )


def test_create_constructor_steering_movement_invalid_behaviour() -> None:
    """Test create_constructor() with an invalid steering movement behaviour."""
    with pytest.raises(
        expected_exception=KeyError,
        match="Test",
    ):
        create_constructor(
            json.dumps(
                {
                    "game_object_type": "Player",
                    "texture_paths": ["floor.png"],
                    "components": {
                        "SteeringMovement": {
                            "Default": ["Test"],
                        },
                    },
                },
            ),
        )


def test_create_constructor_invalid_component() -> None:
    """Test create_constructor() with an invalid component."""
    with pytest.raises(expected_exception=KeyError, match="Test"):
        create_constructor(
            json.dumps(
                {
                    "game_object_type": "Player",
                    "texture_paths": ["floor.png"],
                    "components": {"Test": []},
                },
            ),
        )


def test_create_constructor_kinematic() -> None:
    """Test create_constructor() with a kinematic game object."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "kinematic": True,
            },
        ),
    )
    assert constructor.components[0]


def test_create_constructor_static() -> None:
    """Test create_constructor() with a static game object."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "static": True,
            },
        ),
    )
    assert constructor.components[0]


def test_create_constructor_kinematic_and_static() -> None:
    """Test create_constructor() with a kinematic and static game object."""
    constructor = create_constructor(
        json.dumps(
            {
                "game_object_type": "Player",
                "texture_paths": ["floor.png"],
                "kinematic": True,
                "static": True,
            },
        ),
    )
    assert constructor.components[0]
