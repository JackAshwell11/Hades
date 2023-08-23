"""Tests all classes and functions in game_objects/systems/attributes.py."""
from __future__ import annotations

# Builtin
from typing import ClassVar

# Pip
import pytest

# Custom
from hades.game_objects.components import (
    Armour,
    ArmourRegen,
    ArmourRegenCooldown,
    GameObjectAttributeBase,
    Health,
)
from hades.game_objects.registry import Registry, RegistryError
from hades.game_objects.systems.attributes import (
    ArmourRegenSystem,
    GameObjectAttributeError,
    GameObjectAttributeSystem,
)

__all__ = ()


class FullGameObjectAttribute(GameObjectAttributeBase):
    """Represents a full game object attribute useful for testing."""


class EmptyGameObjectAttribute(GameObjectAttributeBase):
    """Represents an empty game object attribute useful for testing."""

    instant_effect: ClassVar[bool] = False
    maximum: ClassVar[bool] = False
    status_effect: ClassVar[bool] = False
    upgradable: ClassVar[bool] = False


@pytest.fixture()
def registry() -> Registry:
    """Create a registry for use in testing.

    Returns:
        The registry for use in testing.
    """
    return Registry()


@pytest.fixture()
def armour_regen_system(registry: Registry) -> ArmourRegenSystem:
    """Create an armour regen system for use in testing.

    Args:
        registry: The registry for use in testing.

    Returns:
        The armour regen system for use in testing.
    """
    registry.create_game_object(Armour(50, 4), ArmourRegenCooldown(4, 5), ArmourRegen())
    armour_regen_system = ArmourRegenSystem(registry)
    registry.add_system(armour_regen_system)
    return armour_regen_system


@pytest.fixture()
def game_object_attribute_system(registry: Registry) -> GameObjectAttributeSystem:
    """Create a game object attribute system for use in testing.

    Args:
        registry: The registry for use in testing.

    Returns:
        The game object attribute system for use in testing.
    """
    game_object_attribute_system = GameObjectAttributeSystem(registry)
    registry.add_system(game_object_attribute_system)
    return game_object_attribute_system


@pytest.fixture()
def game_object_attribute_system_attributes(
    game_object_attribute_system: GameObjectAttributeSystem,
) -> GameObjectAttributeSystem:
    """Create a game object attribute system with attributes for use in testing.

    Args:
        game_object_attribute_system: The game object attribute system for use in
            testing.

    Returns:
        The game object attribute system with attributes for use in testing.
    """
    game_object_attribute_system.registry.create_game_object(
        FullGameObjectAttribute(150, 3),
        EmptyGameObjectAttribute(100, 5),
    )
    game_object_attribute_system.GAME_OBJECT_ATTRIBUTES.add(FullGameObjectAttribute)
    game_object_attribute_system.GAME_OBJECT_ATTRIBUTES.add(EmptyGameObjectAttribute)
    return game_object_attribute_system


@pytest.fixture()
def game_object_attribute_system_health_armour(
    game_object_attribute_system: GameObjectAttributeSystem,
) -> GameObjectAttributeSystem:
    """Create a game object attribute system with health and armour for use in testing.

    Args:
        game_object_attribute_system: The game object attribute system for use in
            testing.
    """
    game_object_attribute_system.registry.create_game_object(
        Health(300, 4),
        Armour(100, 6),
    )
    game_object_attribute_system.GAME_OBJECT_ATTRIBUTES.add(Health)
    game_object_attribute_system.GAME_OBJECT_ATTRIBUTES.add(Armour)
    return game_object_attribute_system


def test_raise_game_object_attribute_error() -> None:
    """Test that GameObjectAttributeError is raised correctly."""
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match="The game object attribute `test` cannot be upgraded.",
    ):
        raise GameObjectAttributeError(name="test", error="be upgraded")


def test_armour_regen_system_init(armour_regen_system: ArmourRegenSystem) -> None:
    """Test that the armour regen system is initialised correctly.

    Args:
        armour_regen_system: The armour regen system for use in testing.
    """
    assert (
        repr(armour_regen_system)
        == "<ArmourRegenSystem (Description=`Provides facilities to manipulate armour"
        " regen components.`)>"
    )


def test_armour_regen_system_update_small_deltatime(
    armour_regen_system: ArmourRegenSystem,
) -> None:
    """Test that the armour regen component is updated with a small delta time.

    Args:
        armour_regen_system: The armour regen system for use in testing.
    """
    armour = armour_regen_system.registry.get_component_for_game_object(0, Armour)
    armour.value -= 10
    armour_regen_system.update(2)
    assert armour.value == 40
    assert (
        armour_regen_system.registry.get_component_for_game_object(
            0,
            ArmourRegen,
        ).time_since_armour_regen
        == 2
    )


def test_armour_regen_system_update_large_deltatime(
    armour_regen_system: ArmourRegenSystem,
) -> None:
    """Test that the armour regen component is updated with a large deltatime.

    Args:
        armour_regen_system: The armour regen system for use in testing.
    """
    armour = armour_regen_system.registry.get_component_for_game_object(0, Armour)
    armour.value -= 10
    armour_regen_system.update(6)
    assert armour.value == 41
    assert (
        armour_regen_system.registry.get_component_for_game_object(
            0,
            ArmourRegen,
        ).time_since_armour_regen
        == 0
    )


def test_armour_regen_system_update_multiple_updates(
    armour_regen_system: ArmourRegenSystem,
) -> None:
    """Test that the armour regen component is updated correctly multiple times.

    Args:
        armour_regen_system: The armour regen system for use in testing.
    """
    armour, armour_regen = armour_regen_system.registry.get_component_for_game_object(
        0,
        Armour,
    ), armour_regen_system.registry.get_component_for_game_object(0, ArmourRegen)
    armour.value -= 10
    armour_regen_system.update(1)
    assert armour.value == 40
    assert armour_regen.time_since_armour_regen == 1
    armour_regen_system.update(2)
    assert armour.value == 40
    assert armour_regen.time_since_armour_regen == 3


def test_armour_regen_system_update_full_armour(
    armour_regen_system: ArmourRegenSystem,
) -> None:
    """Test that the armour regen component is updated even when armour is already full.

    Args:
        armour_regen_system: The armour regen system for use in testing.
    """
    armour_regen_system.update(5)
    assert (
        armour_regen_system.registry.get_component_for_game_object(0, Armour).value
        == 50
    )
    assert (
        armour_regen_system.registry.get_component_for_game_object(
            0,
            ArmourRegen,
        ).time_since_armour_regen
        == 0
    )


def test_game_object_attribute_system_init(
    game_object_attribute_system: GameObjectAttributeSystem,
) -> None:
    """Test that the game object attribute system is initialised correctly.

    Args:
        game_object_attribute_system: The game object attribute system for use in
            testing.
    """
    assert (
        repr(game_object_attribute_system)
        == "<GameObjectAttributeSystem (Description=`Provides facilities to manipulate"
        " game object attributes.`)>"
    )


def test_game_object_attribute_system_update_no_deltatime(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a game object is updated when no time has passed.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    game_object_attribute_system_attributes.apply_status_effect(
        0,
        FullGameObjectAttribute,
        (lambda level: level * 2, lambda level: level + 100),
        2,
    )
    game_object_attribute_system_attributes.update(0)
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    assert full_game_object_attribute.applied_status_effect
    assert full_game_object_attribute.applied_status_effect.time_counter == 0


def test_game_object_attribute_system_update_larger_deltatime(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a status effect is removed if delta time is larger than the duration.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    game_object_attribute_system_attributes.apply_status_effect(
        0,
        FullGameObjectAttribute,
        (lambda level: 2**level, lambda _: 20),
        2,
    )
    game_object_attribute_system_attributes.update(30)
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    assert full_game_object_attribute.value == 150
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_game_object_attribute_system_update_multiple_deltatimes(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a status effect is removed after multiple updates.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    # Apply the status effect and check the attribute's state after one update
    game_object_attribute_system_attributes.apply_status_effect(
        0,
        FullGameObjectAttribute,
        (lambda level: 50 + level / 2, lambda level: 3**level + 50),
        3,
    )
    game_object_attribute_system_attributes.update(40)
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    assert full_game_object_attribute.applied_status_effect
    assert full_game_object_attribute.applied_status_effect.time_counter == 40
    assert full_game_object_attribute.value == 201.5
    assert full_game_object_attribute.max_value == 201.5

    # Check the attribute's state after the final update
    game_object_attribute_system_attributes.update(40)
    assert full_game_object_attribute.value == 150
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_game_object_attribute_system_update_less_than_original(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a status effect is removed when value is less than the original.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    game_object_attribute_system_attributes.apply_status_effect(
        0,
        FullGameObjectAttribute,
        (lambda _: 100, lambda _: 20),
        4,
    )
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    full_game_object_attribute.value -= 200
    game_object_attribute_system_attributes.update(30)
    assert full_game_object_attribute.value == 50
    assert full_game_object_attribute.max_value == 150
    assert not full_game_object_attribute.applied_status_effect


def test_game_object_attribute_system_update_no_status_effect(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a game object is updated if a status effect does not exist.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    assert not full_game_object_attribute.applied_status_effect
    game_object_attribute_system_attributes.update(5)
    assert not full_game_object_attribute.applied_status_effect


def test_game_object_attribute_system_update_not_allowed(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that updating a game object fails on an empty game object attribute.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    empty_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            EmptyGameObjectAttribute,
        )
    )
    assert not empty_game_object_attribute.applied_status_effect
    game_object_attribute_system_attributes.update(1)
    assert not empty_game_object_attribute.applied_status_effect


def test_game_object_attribute_system_upgrade_value_equal_max(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a full game object attribute is upgraded correctly.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    assert game_object_attribute_system_attributes.upgrade(
        0,
        FullGameObjectAttribute,
        lambda level: 150 * (level + 1),
    )
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    assert full_game_object_attribute.value == 300
    assert full_game_object_attribute.max_value == 300
    assert full_game_object_attribute.current_level == 1


def test_game_object_attribute_system_upgrade_value_lower_max(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a full game object attribute is upgraded if value is lower than max.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    full_game_object_attribute.value -= 50
    assert game_object_attribute_system_attributes.upgrade(
        0,
        FullGameObjectAttribute,
        lambda level: 150 + 2 ^ level,
    )
    assert full_game_object_attribute.value == 101
    assert full_game_object_attribute.max_value == 151
    assert full_game_object_attribute.current_level == 1


def test_game_object_attribute_system_upgrade_max_limit(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a full game object attribute is not upgraded if level limit is reached.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    game_object_attribute_system_attributes.upgrade(
        0,
        FullGameObjectAttribute,
        lambda _: 0,
    )
    game_object_attribute_system_attributes.upgrade(
        0,
        FullGameObjectAttribute,
        lambda _: 0,
    )
    game_object_attribute_system_attributes.upgrade(
        0,
        FullGameObjectAttribute,
        lambda _: 0,
    )
    assert not game_object_attribute_system_attributes.upgrade(
        0,
        FullGameObjectAttribute,
        lambda _: 0,
    )


def test_game_object_attribute_system_upgrade_invalid_increase(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a full game object attribute is not upgraded given an invalid increase.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    with pytest.raises(expected_exception=TypeError):
        game_object_attribute_system_attributes.upgrade(
            0,
            FullGameObjectAttribute,
            lambda _: "str",  # type: ignore[arg-type,return-value]
        )


def test_game_object_attribute_system_upgrade_not_allowed(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that an empty game object attribute raises an error when upgraded.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot be upgraded."
        ),
    ):
        game_object_attribute_system_attributes.upgrade(
            0,
            EmptyGameObjectAttribute,
            lambda _: 0,
        )


def test_game_object_attribute_system_upgrade_invalid_game_object_id(
    game_object_attribute_system: GameObjectAttributeSystem,
) -> None:
    """Test that an empty game object attribute raises an error when upgraded.

    Args:
        game_object_attribute_system: The game object attribute system for use in
            testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        game_object_attribute_system.upgrade(-1, EmptyGameObjectAttribute, lambda _: 0)


def test_game_object_attribute_system_apply_instant_effect_lower(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that an instant effect is applied if the value is lower than the max.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    full_game_object_attribute.value -= 50
    assert game_object_attribute_system_attributes.apply_instant_effect(
        0,
        FullGameObjectAttribute,
        lambda level: 10 * level,
        2,
    )
    assert full_game_object_attribute.value == 120


def test_game_object_attribute_system_apply_instant_effect_equal(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that an instant effect is not applied if the value is equal to the max.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    assert not game_object_attribute_system_attributes.apply_instant_effect(
        0,
        FullGameObjectAttribute,
        lambda _: 50,
        3,
    )
    assert (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        ).value
        == 150
    )


def test_game_object_attribute_system_apply_instant_effect_not_allowed(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that an instant effect raises an error on an empty game object attribute.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot have an"
            " instant effect."
        ),
    ):
        game_object_attribute_system_attributes.apply_instant_effect(
            0,
            EmptyGameObjectAttribute,
            lambda _: 0,
            5,
        )


def test_game_object_attribute_system_apply_instant_effect_invalid_game_object_id(
    game_object_attribute_system: GameObjectAttributeSystem,
) -> None:
    """Test that an empty game object attribute raises an error when upgraded.

    Args:
        game_object_attribute_system: The game object attribute system for use in
            testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        game_object_attribute_system.apply_instant_effect(
            -1,
            EmptyGameObjectAttribute,
            lambda _: 0,
            0,
        )


def test_game_object_attribute_system_apply_status_effect_no_applied_effect(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a status effect is applied if no status effect is currently applied.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    assert game_object_attribute_system_attributes.apply_status_effect(
        0,
        FullGameObjectAttribute,
        (lambda level: 150 + 3**level, lambda level: 20 + 10 * level),
        2,
    )
    applied_status_effect = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        ).applied_status_effect
    )
    assert (
        repr(applied_status_effect)
        == "StatusEffect(value=159, duration=40, original_value=150,"
        " original_max_value=150, time_counter=0)"
    )


def test_game_object_attribute_system_apply_status_effect_value_lower_max(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a status effect is applied if the value is lower than the max.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    full_game_object_attribute = (
        game_object_attribute_system_attributes.registry.get_component_for_game_object(
            0,
            FullGameObjectAttribute,
        )
    )
    full_game_object_attribute.value -= 20
    assert game_object_attribute_system_attributes.apply_status_effect(
        0,
        FullGameObjectAttribute,
        (lambda level: 20 * level, lambda level: 10 - 2**level),
        3,
    )
    assert (
        repr(full_game_object_attribute.applied_status_effect)
        == "StatusEffect(value=60, duration=2, original_value=130,"
        " original_max_value=150, time_counter=0)"
    )


def test_game_object_attribute_system_apply_status_effect_existing_status_effect(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a status effect is not applied if a status effect is already applied.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    game_object_attribute_system_attributes.apply_status_effect(
        0,
        FullGameObjectAttribute,
        (lambda _: 50, lambda _: 20),
        3,
    )
    assert not game_object_attribute_system_attributes.apply_status_effect(
        0,
        FullGameObjectAttribute,
        (lambda _: 60, lambda _: 30),
        2,
    )


def test_game_object_attribute_system_apply_status_effect_not_allowed(
    game_object_attribute_system_attributes: GameObjectAttributeSystem,
) -> None:
    """Test that a status effect raises an error on an empty game object attribute.

    Args:
        game_object_attribute_system_attributes: The game object attribute system with
            attributes for use in testing.
    """
    with pytest.raises(
        expected_exception=GameObjectAttributeError,
        match=(
            "The game object attribute `EmptyGameObjectAttribute` cannot have a status"
            " effect."
        ),
    ):
        game_object_attribute_system_attributes.apply_status_effect(
            0,
            EmptyGameObjectAttribute,
            (lambda _: 0, lambda _: 0),
            6,
        )


def test_game_object_attribute_system_apply_status_effect_invalid_game_object_id(
    game_object_attribute_system: GameObjectAttributeSystem,
) -> None:
    """Test that an empty game object attribute raises an error when upgraded.

    Args:
        game_object_attribute_system: The game object attribute system for use in
            testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        game_object_attribute_system.apply_status_effect(
            -1,
            EmptyGameObjectAttribute,
            (lambda _: 0, lambda _: 0),
            0,
        )


def test_game_object_attribute_system_deal_damage_low_health_armour(
    game_object_attribute_system_health_armour: GameObjectAttributeSystem,
) -> None:
    """Test that damage is dealt when health and armour are lower than damage.

    Args:
        game_object_attribute_system_health_armour: The game object attribute system
            with health and armour for use in testing.
    """
    registry = game_object_attribute_system_health_armour.registry
    game_object_attribute_system_health_armour.deal_damage(0, 350)
    assert registry.get_component_for_game_object(0, Health).value == 50
    assert registry.get_component_for_game_object(0, Armour).value == 0


def test_game_object_attribute_system_deal_damage_large_armour(
    game_object_attribute_system_health_armour: GameObjectAttributeSystem,
) -> None:
    """Test that no damage is dealt when armour is larger than damage.

    Args:
        game_object_attribute_system_health_armour: The game object attribute system
            with health and armour for use in testing.
    """
    registry = game_object_attribute_system_health_armour.registry
    game_object_attribute_system_health_armour.deal_damage(0, 50)
    assert registry.get_component_for_game_object(0, Health).value == 300
    assert registry.get_component_for_game_object(0, Armour).value == 50


def test_game_object_attribute_system_deal_damage_zero_damage(
    game_object_attribute_system_health_armour: GameObjectAttributeSystem,
) -> None:
    """Test that no damage is dealt when damage is zero.

    Args:
        game_object_attribute_system_health_armour: The game object attribute system
            with health and armour for use in testing.
    """
    registry = game_object_attribute_system_health_armour.registry
    game_object_attribute_system_health_armour.deal_damage(0, 0)
    assert registry.get_component_for_game_object(0, Health).value == 300
    assert registry.get_component_for_game_object(0, Armour).value == 100


def test_game_object_attribute_system_deal_damage_zero_armour(
    game_object_attribute_system_health_armour: GameObjectAttributeSystem,
) -> None:
    """Test that damage is dealt when armour is zero.

    Args:
        game_object_attribute_system_health_armour: The game object attribute system
            with health and armour for use in testing.
    """
    registry = game_object_attribute_system_health_armour.registry
    armour = registry.get_component_for_game_object(0, Armour)
    armour.value = 0
    game_object_attribute_system_health_armour.deal_damage(0, 100)
    assert registry.get_component_for_game_object(0, Health).value == 200
    assert armour.value == 0


def test_game_object_attribute_system_deal_damage_zero_health(
    game_object_attribute_system_health_armour: GameObjectAttributeSystem,
) -> None:
    """Test that damage is dealt when health is zero.

    Args:
        game_object_attribute_system_health_armour: The game object attribute system
            with health and armour for use in testing.
    """
    registry = game_object_attribute_system_health_armour.registry
    health = registry.get_component_for_game_object(0, Health)
    health.value = 0
    game_object_attribute_system_health_armour.deal_damage(0, 50)
    assert health.value == 0
    assert registry.get_component_for_game_object(0, Armour).value == 50


def test_game_object_attribute_system_deal_damage_nonexistent_attributes(
    game_object_attribute_system: GameObjectAttributeSystem,
) -> None:
    """Test that no damage is dealt when the attributes are not initialised.

    Args:
        game_object_attribute_system: The game object attribute system for use in
            testing.
    """
    game_object_attribute_system.registry.create_game_object()
    with pytest.raises(expected_exception=KeyError):
        game_object_attribute_system.deal_damage(0, 100)


def test_game_object_attribute_system_deal_damage_invalid_game_object_id(
    game_object_attribute_system: GameObjectAttributeSystem,
) -> None:
    """Test that an empty game object attribute raises an error when upgraded.

    Args:
        game_object_attribute_system: The game object attribute system for use in
            testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        game_object_attribute_system.deal_damage(-1, 100)
