"""Tests all functions in game_objects/attributes.py."""
from __future__ import annotations

# Builtin
from typing import cast

# Pip
import pytest

# Custom
from hades.game_objects.attributes import (
    GameObjectAttributeBase,
    GameObjectAttributeError,
    deal_damage,
)
from hades.game_objects.base import ComponentType
from hades.game_objects.system import ECS

__all__ = ()


class FullGameObjectAttribute(GameObjectAttributeBase):
    """Represents a full game object attribute useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.HEALTH


class EmptyGameObjectAttribute(GameObjectAttributeBase):
    """Represents an empty game object attribute useful for testing."""

    # Class variables
    component_type: ComponentType = ComponentType.ARMOUR
    instant_effect: bool = False
    maximum: bool = False
    status_effect: bool = False
    upgradable: bool = False


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    return ECS()


@pytest.fixture()
def full_game_object_attribute(ecs: ECS) -> FullGameObjectAttribute:
    """Create a full game object attribute for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The full game object attribute for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.HEALTH: (150, 3)}},
        FullGameObjectAttribute,
    )
    return cast(
        FullGameObjectAttribute,
        ecs.get_component_for_game_object(0, ComponentType.HEALTH),
    )


@pytest.fixture()
def empty_game_object_attribute(ecs: ECS) -> EmptyGameObjectAttribute:
    """Create an empty game object attribute for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The empty game object attribute for use in testing.
    """
    ecs.add_game_object(
        {"attributes": {ComponentType.ARMOUR: (100, 5)}},
        EmptyGameObjectAttribute,
    )
    return cast(
        EmptyGameObjectAttribute,
        ecs.get_component_for_game_object(0, ComponentType.ARMOUR),
    )


@pytest.fixture()
def both_game_object_attributes(
    ecs: ECS,
) -> tuple[FullGameObjectAttribute, EmptyGameObjectAttribute]:
    """Create two game object attributes for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        Both game object attributes for use in testing.
    """
    ecs.add_game_object(
        {
            "attributes": {
                ComponentType.HEALTH: (200, 4),
                ComponentType.ARMOUR: (150, 6),
            },
        },
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    )
    return cast(
        FullGameObjectAttribute,
        ecs.get_component_for_game_object(0, ComponentType.HEALTH),
    ), cast(
        EmptyGameObjectAttribute,
        ecs.get_component_for_game_object(0, ComponentType.ARMOUR),
    )


def test_raise_game_object_attribute_error() -> None:
    """Test that GameObjectAttributeError is raised correctly."""
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match="The game object attribute `test` cannot be upgraded.",
    ):
        raise GameObjectAttributeError(name="test", error="be upgraded")


def test_full_game_object_attribute_init(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is initialised correctly.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert (
        repr(full_game_object_attribute)
        == "<FullGameObjectAttribute (Value=150) (Max value=150) (Level=0/3)>"
    )


def test_empty_game_object_attribute_init(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that an empty game object attribute is initialised correctly.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    assert (
        repr(empty_game_object_attribute)
        == "<EmptyGameObjectAttribute (Value=100) (Max value=inf) (Level=0/5)>"
    )


def test_full_game_object_attribute_setter_lower(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is set with a lower value correctly.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value = 100
    assert full_game_object_attribute.value == 100


def test_full_game_object_attribute_setter_higher(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is set with a higher value correctly.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value = 200
    assert full_game_object_attribute.value == 150


def test_full_game_object_attribute_setter_isub(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that subtracting a value from the full game object attribute is correct.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value -= 200
    assert full_game_object_attribute.value == 0


def test_full_game_object_attribute_setter_iadd(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that adding a value to the full game object attribute is correct.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value += 100
    assert full_game_object_attribute.value == 150


def test_full_game_object_attribute_upgrade_value_equal_max(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is upgraded correctly.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert full_game_object_attribute.upgrade(lambda level: 150 * (level + 1))
    assert full_game_object_attribute.value == 300
    assert full_game_object_attribute.max_value == 300
    assert full_game_object_attribute.current_level == 1


def test_full_game_object_attribute_upgrade_value_lower_max(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is upgraded if value is lower than max.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value -= 50
    assert full_game_object_attribute.upgrade(lambda level: 150 + 2 ^ level)
    assert full_game_object_attribute.value == 101
    assert full_game_object_attribute.max_value == 151
    assert full_game_object_attribute.current_level == 1


def test_full_game_object_attribute_upgrade_max_limit(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is not upgraded if level limit is reached.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.upgrade(lambda _: 0)
    full_game_object_attribute.upgrade(lambda _: 0)
    full_game_object_attribute.upgrade(lambda _: 0)
    assert not full_game_object_attribute.upgrade(lambda _: 0)


def test_full_game_object_attribute_upgrade_invalid_increase(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a full game object attribute is not upgraded given an invalid increase.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    with pytest.raises(expected_exception=TypeError):
        full_game_object_attribute.upgrade(
            lambda _: "str",  # type: ignore[arg-type,return-value]
        )


def test_full_game_object_attribute_apply_instant_effect_lower(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that an instant effect is applied if the value is lower than the max.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value -= 50
    assert full_game_object_attribute.apply_instant_effect(lambda level: 10 * level, 2)
    assert full_game_object_attribute.value == 120


def test_full_game_object_attribute_apply_instant_effect_equal(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that an instant effect is not applied if the value is equal to the max.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert not full_game_object_attribute.apply_instant_effect(lambda _: 50, 3)
    assert full_game_object_attribute.value == 150


def test_full_game_object_attribute_apply_status_effect_no_applied_effect(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is applied if no status effect is currently applied.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert full_game_object_attribute.apply_status_effect(
        lambda level: 150 + 3**level,
        lambda level: 20 + 10 * level,
        2,
    )
    assert (
        repr(full_game_object_attribute.applied_status_effect)
        == "StatusEffect(value=159, duration=40, original_value=150,"
        " original_max_value=150, time_counter=0)"
    )


def test_full_game_object_attribute_apply_status_effect_value_lower_max(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is applied if the value is lower than the max.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.value -= 20
    assert full_game_object_attribute.apply_status_effect(
        lambda level: 20 * level,
        lambda level: 10 - 2**level,
        3,
    )
    assert (
        repr(full_game_object_attribute.applied_status_effect)
        == "StatusEffect(value=60, duration=2, original_value=130,"
        " original_max_value=150, time_counter=0)"
    )


def test_full_game_object_attribute_apply_status_effect_existing_status_effect(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is not applied if a status effect is already applied.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.apply_status_effect(lambda _: 50, lambda _: 20, 3)
    assert not full_game_object_attribute.apply_status_effect(
        lambda _: 60,
        lambda _: 30,
        2,
    )


def test_full_game_object_attribute_on_update_no_deltatime(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a game object is updated when no time has passed.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.apply_status_effect(
        lambda level: level * 2,
        lambda level: level + 100,
        2,
    )
    full_game_object_attribute.on_update(0)
    assert full_game_object_attribute.applied_status_effect
    assert full_game_object_attribute.applied_status_effect.time_counter == 0


def test_full_game_object_attribute_on_update_larger_deltatime(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is removed if deltatime is larger than the duration.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.apply_status_effect(
        lambda level: 2**level,
        lambda _: 20,
        2,
    )
    full_game_object_attribute.on_update(30)
    assert full_game_object_attribute.value == 150
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_full_game_object_attribute_on_update_multiple_deltatimes(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is removed after multiple updates.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    # Apply the status effect and check the attribute's state after one update
    full_game_object_attribute.apply_status_effect(
        lambda level: 50 + level / 2,
        lambda level: 3**level + 50,
        3,
    )
    full_game_object_attribute.on_update(40)
    assert full_game_object_attribute.applied_status_effect
    assert full_game_object_attribute.applied_status_effect.time_counter == 40
    assert full_game_object_attribute.value == 201.5
    assert full_game_object_attribute.max_value == 201.5

    # Check the attribute's state after the final update
    full_game_object_attribute.on_update(40)
    assert full_game_object_attribute.value == 150
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_full_game_object_attribute_on_update_less_than_original(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a status effect is removed when value is less than the original.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    full_game_object_attribute.apply_status_effect(lambda _: 100, lambda _: 20, 4)
    full_game_object_attribute.value -= 200
    full_game_object_attribute.on_update(30)
    assert full_game_object_attribute.value == 50
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_full_game_object_attribute_on_update_no_status_effect(
    full_game_object_attribute: FullGameObjectAttribute,
) -> None:
    """Test that a game object is updated if a status effect does not exist.

    Args:
        full_game_object_attribute: The full game object attribute for use in testing.
    """
    assert not full_game_object_attribute.applied_status_effect
    full_game_object_attribute.on_update(5)
    assert not full_game_object_attribute.applied_status_effect


def test_empty_game_object_attribute_setter(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that an empty game object attribute's value can be changed.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    empty_game_object_attribute.value -= 10
    assert empty_game_object_attribute.value == 90


def test_empty_game_object_attribute_upgrade(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that an empty game object attribute raises an error when upgraded.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot be upgraded."
        ),
    ):
        empty_game_object_attribute.upgrade(lambda _: 0)


def test_empty_game_object_attribute_instant_effect(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that an instant effect raises an error on an empty game object attribute.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot have an"
            " instant effect."
        ),
    ):
        empty_game_object_attribute.apply_instant_effect(lambda _: 0, 5)


def test_empty_game_object_attribute_apply_status_effect(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that a status effect raises an error on an empty game object attribute.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot have a status"
            " effect."
        ),
    ):
        empty_game_object_attribute.apply_status_effect(lambda _: 0, lambda _: 0, 6)


def test_empty_game_object_attribute_on_update(
    empty_game_object_attribute: EmptyGameObjectAttribute,
) -> None:
    """Test that updating a game object fails on an empty game object attribute.

    Args:
        empty_game_object_attribute: The empty game object attribute for use in testing.
    """
    assert not empty_game_object_attribute.applied_status_effect
    empty_game_object_attribute.on_update(1)
    assert not empty_game_object_attribute.applied_status_effect


def test_deal_damage_low_health_armour(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that damage is dealt when health and armour are lower than damage.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    both_game_object_attributes[0].value = 150
    both_game_object_attributes[1].value = 100
    deal_damage(0, ecs, 200)
    assert both_game_object_attributes[0].value == 50
    assert both_game_object_attributes[1].value == 0


def test_deal_damage_large_armour(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that no damage is dealt when armour is larger than damage.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    deal_damage(0, ecs, 100)
    assert both_game_object_attributes[0].value == 200
    assert both_game_object_attributes[1].value == 50


def test_deal_damage_zero_damage(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that no damage is dealt when damage is zero.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    deal_damage(0, ecs, 0)
    assert both_game_object_attributes[0].value == 200
    assert both_game_object_attributes[1].value == 150


def test_deal_damage_zero_armour(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that damage is dealt when armour is zero.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    both_game_object_attributes[1].value = 0
    deal_damage(0, ecs, 100)
    assert both_game_object_attributes[0].value == 100
    assert both_game_object_attributes[1].value == 0


def test_deal_damage_zero_health(
    ecs: ECS,
    both_game_object_attributes: tuple[
        FullGameObjectAttribute,
        EmptyGameObjectAttribute,
    ],
) -> None:
    """Test that damage is dealt when health is zero.

    Args:
        ecs: The entity component system for use in testing.
        both_game_object_attributes: Both game object attributes for use in testing.
    """
    both_game_object_attributes[0].value = 0
    deal_damage(0, ecs, 200)
    assert both_game_object_attributes[0].value == 0
    assert both_game_object_attributes[1].value == 0


def test_deal_damage_nonexistent_attributes(ecs: ECS) -> None:
    """Test that no damage is dealt when the attributes are not initialised.

    Args:
        ecs: The entity component system for use in testing.
    """
    ecs.add_game_object({})
    with pytest.raises(expected_exception=KeyError):
        deal_damage(0, ecs, 100)
