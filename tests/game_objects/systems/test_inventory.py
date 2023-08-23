"""Tests all classes and functions in game_objects/systems/inventory.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.game_objects.components import Inventory
from hades.game_objects.registry import Registry, RegistryError
from hades.game_objects.systems.inventory import InventorySpaceError, InventorySystem

__all__ = ()


@pytest.fixture()
def registry() -> Registry:
    """Create a registry for use in testing.

    Returns:
        The registry for use in testing.
    """
    return Registry()


@pytest.fixture()
def inventory_system(registry: Registry) -> InventorySystem:
    """Create an inventory system for use in testing.

    Args:
        registry: The registry for use in testing.

    Returns:
        The inventory system for use in testing.
    """
    registry.create_game_object(Inventory(3, 6))
    inventory_system = InventorySystem(registry)
    registry.add_system(inventory_system)
    return inventory_system


def test_raise_inventory_space_error_full() -> None:
    """Test that InventorySpaceError is raised correctly when full."""
    with pytest.raises(
        expected_exception=InventorySpaceError,
        match="The inventory is full.",
    ):
        raise InventorySpaceError(full=True)


def test_raise_inventory_space_error_empty() -> None:
    """Test that InventorySpaceError is raised correctly when empty."""
    with pytest.raises(
        expected_exception=InventorySpaceError,
        match="The inventory is empty.",
    ):
        raise InventorySpaceError(full=False)


def test_inventory_system_init(inventory_system: InventorySystem) -> None:
    """Test that the inventory system is initialised correctly.

    Args:
        inventory_system: The inventory system for use in testing.
    """
    assert (
        repr(inventory_system)
        == "<InventorySystem (Description=`Provides facilities to manipulate inventory"
        " components.`)>"
    )


def test_inventory_system_add_item_to_inventory_valid(
    inventory_system: InventorySystem,
) -> None:
    """Test that a valid item is added to the inventory correctly.

    Args:
        inventory_system: The inventory system for use in testing.
    """
    inventory_system.add_item_to_inventory(0, 50)
    assert inventory_system.registry.get_component_for_game_object(
        0,
        Inventory,
    ).inventory == [50]


def test_inventory_system_add_item_to_inventory_zero_size(
    inventory_system: InventorySystem,
) -> None:
    """Test that a valid item is not added to a zero size inventory.

    Args:
        inventory_system: The inventory system for use in testing.
    """
    inventory_system.registry.get_component_for_game_object(0, Inventory).width = 0
    with pytest.raises(expected_exception=InventorySpaceError):
        inventory_system.add_item_to_inventory(0, 50)


def test_inventory_system_add_item_to_inventory_invalid_game_object_id(
    inventory_system: InventorySystem,
) -> None:
    """Test that an exception is raised if an invalid game object ID is provided.

    Args:
        inventory_system: The inventory system for use in testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        inventory_system.add_item_to_inventory(-1, 50)


def test_inventory_system_remove_item_from_inventory_valid(
    inventory_system: InventorySystem,
) -> None:
    """Test that a valid item is removed from the inventory correctly.

    Args:
        inventory_system: The inventory system for use in testing.
    """
    inventory_system.add_item_to_inventory(0, 1)
    inventory_system.add_item_to_inventory(0, 7)
    inventory_system.add_item_to_inventory(0, 4)
    assert inventory_system.remove_item_from_inventory(0, 1) == 7
    assert inventory_system.registry.get_component_for_game_object(
        0,
        Inventory,
    ).inventory == [1, 4]


def test_inventory_system_remove_item_from_inventory_large_index(
    inventory_system: InventorySystem,
) -> None:
    """Test that an exception is raised if a larger index is provided.

    Args:
        inventory_system: The inventory system for use in testing.
    """
    inventory_system.add_item_to_inventory(0, 5)
    inventory_system.add_item_to_inventory(0, 10)
    inventory_system.add_item_to_inventory(0, 50)
    with pytest.raises(expected_exception=InventorySpaceError):
        inventory_system.remove_item_from_inventory(0, 10)


def test_inventory_system_remove_item_from_inventory_invalid_game_object_id(
    inventory_system: InventorySystem,
) -> None:
    """Test that an exception is raised if an invalid game object ID is provided.

    Args:
        inventory_system: The inventory system for use in testing.
    """
    with pytest.raises(
        expected_exception=RegistryError,
        match="The game object ID `-1` is not registered with the registry.",
    ):
        inventory_system.remove_item_from_inventory(-1, 0)
