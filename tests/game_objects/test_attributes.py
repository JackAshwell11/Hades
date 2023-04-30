"""Tests all functions in game_objects/attributes.py."""
from __future__ import annotations

# Pip
import pytest
from arcade import PymunkPhysicsEngine

# Custom
from hades.game_objects.attributes import (
    EntityAttributeBase,
    EntityAttributeError,
)

__all__ = ()


class FullEntityAttribute(EntityAttributeBase):
    """Represents a full entity attribute useful for testing."""


class EmptyEntityAttribute(EntityAttributeBase):
    """Represents an empty entity attribute useful for testing."""

    # Class variables
    instant_effect: bool = False
    maximum: bool = False
    status_effect: bool = False
    upgradable: bool = False


class PhysicsEngine(PymunkPhysicsEngine):
    """Represents a Pymunk physics engine useful for testing."""


@pytest.fixture()
def full_entity_attribute() -> FullEntityAttribute:
    """Create a full entity attribute for use in testing.

    Returns:
        The full entity attribute for use in testing.
    """
    return FullEntityAttribute(initial_value=150, level_limit=3)


@pytest.fixture()
def empty_entity_attribute() -> EmptyEntityAttribute:
    """Create an empty entity attribute for use in testing.

    Returns:
        The empty entity attribute for use in testing.
    """
    return EmptyEntityAttribute(initial_value=20, level_limit=5)


def test_raise_entity_attribute_error() -> None:
    """Test that EntityAttributeError is raised correctly."""
    with pytest.raises(
        expected_exception=EntityAttributeError,
        match="The entity attribute `test` cannot be upgraded.",
    ):
        raise EntityAttributeError(name="test", error="be upgraded")


def test_full_entity_attribute_init(full_entity_attribute: FullEntityAttribute) -> None:
    """Test that a full entity attribute is initialised correctly.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    assert (
        repr(full_entity_attribute)
        == "<FullEntityAttribute (Value=150) (Max value=150) (Level=0/3)>"
    )


def test_empty_entity_attribute_init(
    empty_entity_attribute: EmptyEntityAttribute,
) -> None:
    """Test that an empty entity attribute is initialised correctly.

    Args:
        empty_entity_attribute: The empty entity attribute for use in testing.
    """
    assert (
        repr(empty_entity_attribute)
        == "<EmptyEntityAttribute (Value=20) (Max value=inf) (Level=0/5)>"
    )


def test_full_entity_attribute_setter_lower(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a full entity attribute is set with a lower value correctly.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.value = 100
    assert full_entity_attribute.value == 100


def test_full_entity_attribute_setter_higher(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a full entity attribute is set with a higher value correctly.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.value = 200
    assert full_entity_attribute.value == 150


def test_full_entity_attribute_setter_isub(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that subtracting a value from the full entity attribute is correct.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.value -= 200
    assert full_entity_attribute.value == 0


def test_full_entity_attribute_setter_iadd(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that adding a value to the full entity attribute is correct.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.value += 100
    assert full_entity_attribute.value == 150


def test_full_entity_attribute_upgrade_value_equal_max(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a full entity attribute is upgraded correctly.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    assert full_entity_attribute.upgrade(lambda level: 150 * (level + 1))
    assert full_entity_attribute.value == 300
    assert full_entity_attribute.max_value == 300
    assert full_entity_attribute.current_level == 1


def test_full_entity_attribute_upgrade_value_lower_max(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a full entity attribute is upgraded if the value is lower than the max.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.value -= 50
    assert full_entity_attribute.upgrade(lambda level: 150 + 2 ^ level)
    assert full_entity_attribute.value == 101
    assert full_entity_attribute.max_value == 151
    assert full_entity_attribute.current_level == 1


def test_full_entity_attribute_upgrade_max_limit(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a full entity attribute is not upgraded if the level limit is reached.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.upgrade(lambda _: 0)
    full_entity_attribute.upgrade(lambda _: 0)
    full_entity_attribute.upgrade(lambda _: 0)
    assert not full_entity_attribute.upgrade(lambda _: 0)


def test_full_entity_attribute_upgrade_invalid_increase(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a full entity attribute is not upgraded when given an invalid lambda.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """

    def bad_func() -> str:
        return "test"

    with pytest.raises(expected_exception=TypeError):
        full_entity_attribute.upgrade(bad_func)  # type: ignore[arg-type]


def test_full_entity_attribute_apply_instant_effect_lower(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that an instant effect is applied if the value is lower than the max.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.value -= 50
    assert full_entity_attribute.apply_instant_effect(lambda level: 10 * level, 2)
    assert full_entity_attribute.value == 120


def test_full_entity_attribute_apply_instant_effect_equal(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that an instant effect is not applied if the value is equal to the max.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    assert not full_entity_attribute.apply_instant_effect(lambda _: 50, 3)
    assert full_entity_attribute.value == 150


def test_full_entity_attribute_apply_status_effect_no_applied_effect(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a status effect is applied if no status effect is currently applied.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    assert full_entity_attribute.apply_status_effect(
        lambda level: 150 + 3**level,
        lambda level: 20 + 10 * level,
        2,
    )
    assert (
        repr(full_entity_attribute.applied_status_effect)
        == "StatusEffect(value=159, duration=40, original_value=150,"
        " original_max_value=150, time_counter=0)"
    )


def test_full_entity_attribute_apply_status_effect_value_lower_max(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a status effect is applied if the value is lower than the max.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.value -= 20
    assert full_entity_attribute.apply_status_effect(
        lambda level: 20 * level,
        lambda level: 10 - 2**level,
        3,
    )
    assert (
        repr(full_entity_attribute.applied_status_effect)
        == "StatusEffect(value=60, duration=2, original_value=130,"
        " original_max_value=150, time_counter=0)"
    )


def test_full_entity_attribute_apply_status_effect_existing_status_effect(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a status effect is not applied if a status effect is already applied.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.apply_status_effect(lambda _: 50, lambda _: 20, 3)
    assert not full_entity_attribute.apply_status_effect(lambda _: 60, lambda _: 30, 2)


def test_full_entity_attribute_update_status_effect_no_deltatime(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a status effect is updated if no time has passed.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.apply_status_effect(
        lambda level: level * 2,
        lambda level: level + 100,
        2,
    )
    full_entity_attribute.update_status_effect(0)
    assert full_entity_attribute.applied_status_effect.time_counter == 0


def test_full_entity_attribute_update_status_effect_larger_deltatime(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a status effect is removed if deltatime is larger than the duration.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.apply_status_effect(lambda level: 2**level, lambda _: 20, 2)
    full_entity_attribute.update_status_effect(30)
    assert full_entity_attribute.value == 150
    assert full_entity_attribute.max_value == 150
    assert not full_entity_attribute.applied_status_effect


def test_full_entity_attribute_update_status_effect_multiple_deltatimes(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a status effect is removed after multiple updates.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    # Apply the status effect and check the attribute's state after one update
    full_entity_attribute.apply_status_effect(
        lambda level: 50 + level / 2,
        lambda level: 3**level + 50,
        3,
    )
    full_entity_attribute.update_status_effect(40)
    assert full_entity_attribute.applied_status_effect.time_counter == 40
    assert full_entity_attribute.value == 201.5
    assert full_entity_attribute.max_value == 201.5

    # Check the attribute's state after the final update
    full_entity_attribute.update_status_effect(40)
    assert full_entity_attribute.value == 150
    assert full_entity_attribute.max_value == 150
    assert not full_entity_attribute.applied_status_effect


def test_full_entity_attribute_update_status_effect_less_than_original(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a status effect is removed even when value is less than the original.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    full_entity_attribute.apply_status_effect(lambda _: 100, lambda _: 20, 4)
    full_entity_attribute.value -= 200
    full_entity_attribute.update_status_effect(30)
    assert full_entity_attribute.value == 50
    assert full_entity_attribute.max_value == 150
    assert not full_entity_attribute.applied_status_effect


def test_full_entity_attribute_apply_status_effect_no_status_effect(
    full_entity_attribute: FullEntityAttribute,
) -> None:
    """Test that a status effect is not updated if it's not applied.

    Args:
        full_entity_attribute: The full entity attribute for use in testing.
    """
    assert not full_entity_attribute.update_status_effect(5)


def test_empty_entity_attribute_setter(
    empty_entity_attribute: EmptyEntityAttribute,
) -> None:
    """Test that an empty entity attribute's value can be changed.

    Args:
        empty_entity_attribute: The empty entity attribute for use in testing.
    """
    empty_entity_attribute.value -= 10
    assert empty_entity_attribute.value == 10


def test_empty_entity_attribute_upgrade(
    empty_entity_attribute: EmptyEntityAttribute,
) -> None:
    """Test that an empty entity attribute raises an error when upgraded.

    Args:
        empty_entity_attribute: The empty entity attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=EntityAttributeError,
        match="The entity attribute `EmptyEntityAttribute` cannot be upgraded.",
    ):
        empty_entity_attribute.upgrade(lambda _: 0)


def test_empty_entity_attribute_instant_effect(
    empty_entity_attribute: EmptyEntityAttribute,
) -> None:
    """Test that an instant effect raises an error on an empty entity attribute.

    Args:
        empty_entity_attribute: The empty entity attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=EntityAttributeError,
        match=(
            "The entity attribute `EmptyEntityAttribute` cannot have an instant effect."
        ),
    ):
        empty_entity_attribute.apply_instant_effect(lambda _: 0, 5)


def test_empty_entity_attribute_apply_status_effect(
    empty_entity_attribute: EmptyEntityAttribute,
) -> None:
    """Test that a status effect raises an error on an empty entity attribute.

    Args:
        empty_entity_attribute: The empty entity attribute for use in testing.
    """
    with pytest.raises(
        expected_exception=EntityAttributeError,
        match=(
            "The entity attribute `EmptyEntityAttribute` cannot have a status effect."
        ),
    ):
        empty_entity_attribute.apply_status_effect(lambda _: 0, lambda _: 0, 6)


def test_empty_entity_attribute_update_status_effect(
    empty_entity_attribute: EmptyEntityAttribute,
) -> None:
    """Test that updating a status effect fails on an empty entity attribute.

    Args:
        empty_entity_attribute: The empty entity attribute for use in testing.
    """
    assert not empty_entity_attribute.update_status_effect(1)
