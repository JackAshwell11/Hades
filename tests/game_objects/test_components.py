"""Tests all functions in game_objects/components.py."""
from __future__ import annotations

# Builtin
from typing import cast

# Pip
import pytest

# Custom
from hades.game_objects.attributes import Armour, ArmourRegenCooldown
from hades.game_objects.base import ComponentType
from hades.game_objects.components import (
    ArmourRegen,
    InstantEffects,
    Inventory,
    InventorySpaceError,
    StatusEffects,
)
from hades.game_objects.system import ECS

__all__ = ()


@pytest.fixture()
def ecs() -> ECS:
    """Create an entity component system for use in testing.

    Returns:
        The entity component system for use in testing.
    """
    return ECS()


@pytest.fixture()
def armour_regen(ecs: ECS) -> ArmourRegen:
    """Create an armour regen component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The armour regen component for use in testing.
    """
    ecs.add_game_object(
        {
            "attributes": {
                ComponentType.ARMOUR: (50, 3),
                ComponentType.ARMOUR_REGEN_COOLDOWN: (4, 5),
            },
        },
        Armour,
        ArmourRegenCooldown,
        ArmourRegen,
    )
    return cast(
        ArmourRegen,
        ecs.get_component_for_game_object(0, ComponentType.ARMOUR_REGEN),
    )


@pytest.fixture()
def instant_effects(ecs: ECS) -> InstantEffects:
    """Create an instant effects component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The instant effects component for use in testing.
    """
    ecs.add_game_object(
        {"instant_effects": (5, {ComponentType.HEALTH: lambda level: 2**level + 5})},
        InstantEffects,
    )
    return cast(
        InstantEffects,
        ecs.get_component_for_game_object(0, ComponentType.INSTANT_EFFECTS),
    )


@pytest.fixture()
def inventory(ecs: ECS) -> Inventory[int]:
    """Create an inventory component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The inventory component for use in testing.
    """
    ecs.add_game_object(
        {"inventory_size": (3, 6)},
        Inventory,
    )
    return cast(
        Inventory[int],
        ecs.get_component_for_game_object(0, ComponentType.INVENTORY),
    )


@pytest.fixture()
def status_effects(ecs: ECS) -> StatusEffects:
    """Create a status effects component for use in testing.

    Args:
        ecs: The entity component system for use in testing.

    Returns:
        The status effects component for use in testing.
    """
    ecs.add_game_object(
        {
            "status_effects": (
                3,
                {
                    ComponentType.ARMOUR: (
                        lambda level: 3**level + 10,
                        lambda level: 3 * level + 5,
                    ),
                },
            ),
        },
        StatusEffects,
    )
    return cast(
        StatusEffects,
        ecs.get_component_for_game_object(0, ComponentType.STATUS_EFFECTS),
    )


def test_raise_inventory_space_error() -> None:
    """Test that InventorySpaceError is raised correctly."""
    with pytest.raises(
        expected_exception=InventorySpaceError,
        match="The inventory is full.",
    ):
        raise InventorySpaceError(full=True)


def test_armour_regen_init(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is initialised correctly.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    assert repr(armour_regen) == "<ArmourRegen (Time since armour regen=0)>"


def test_armour_regen_on_update_small_deltatime(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is updated with a small deltatime.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    armour_regen.armour.value -= 10
    armour_regen.on_update(2)
    assert armour_regen.armour.value == 40
    assert armour_regen.time_since_armour_regen == 2


def test_armour_regen_on_update_large_deltatime(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is updated with a large deltatime.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    armour_regen.armour.value -= 10
    armour_regen.on_update(6)
    assert armour_regen.armour.value == 41
    assert armour_regen.time_since_armour_regen == 0


def test_armour_regen_on_update_multiple_updates(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is updated correctly multiple times.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    armour_regen.armour.value -= 10
    armour_regen.on_update(1)
    assert armour_regen.armour.value == 40
    assert armour_regen.time_since_armour_regen == 1
    armour_regen.on_update(2)
    assert armour_regen.armour.value == 40
    assert armour_regen.time_since_armour_regen == 3


def test_armour_regen_on_update_full_armour(armour_regen: ArmourRegen) -> None:
    """Test that the armour regen component is updated even when armour is already full.

    Args:
        armour_regen: The armour regen component for use in testing.
    """
    armour_regen.on_update(5)
    assert armour_regen.armour.value == 50
    assert armour_regen.time_since_armour_regen == 0


def test_instant_effects_init(instant_effects: InstantEffects) -> None:
    """Test that the instant effects component is initialised correctly.

    Args:
        instant_effects: The instant effects component for use in testing.
    """
    assert repr(instant_effects) == "<InstantEffects (Level limit=5)>"


def test_inventory_init(inventory: Inventory[int]) -> None:
    """Test that the inventory component is initialised correctly.

    Args:
        inventory: The inventory component for use in testing.
    """
    assert repr(inventory) == "<Inventory (Width=3) (Height=6)>"
    assert not inventory.inventory


def test_inventory_add_item_to_inventory_valid(inventory: Inventory[int]) -> None:
    """Test that a valid item is added to the inventory correctly.

    Args:
        inventory: The inventory component for use in testing.
    """
    inventory.add_item_to_inventory(50)
    assert inventory.inventory == [50]


def test_inventory_add_item_to_inventory_zero_size(inventory: Inventory[int]) -> None:
    """Test that a valid item is not added to a zero size inventory.

    Args:
        inventory: The inventory component for use in testing.
    """
    inventory.width = 0
    with pytest.raises(expected_exception=InventorySpaceError):
        inventory.add_item_to_inventory(5)


def test_inventory_remove_item_from_inventory_valid(inventory: Inventory[int]) -> None:
    """Test that a valid item is removed from the inventory correctly.

    Args:
        inventory: The inventory component for use in testing.
    """
    inventory.add_item_to_inventory(1)
    inventory.add_item_to_inventory(7)
    inventory.add_item_to_inventory(4)
    assert inventory.remove_item_from_inventory(1) == 7
    assert inventory.inventory == [1, 4]


def test_inventory_remove_item_from_inventory_large_index(
    inventory: Inventory[int],
) -> None:
    """Test that an exception is raised if a larger index is provided.

    Args:
        inventory: The inventory component for use in testing.
    """
    inventory.add_item_to_inventory(5)
    inventory.add_item_to_inventory(10)
    inventory.add_item_to_inventory(50)
    with pytest.raises(expected_exception=InventorySpaceError):
        inventory.remove_item_from_inventory(10)


def test_status_effects_init(status_effects: StatusEffects) -> None:
    """Test that the status effects component is initialised correctly.

    Args:
        status_effects: The status effects component for use in testing.
    """
    assert repr(status_effects) == "<StatusEffects (Level limit=3)>"
