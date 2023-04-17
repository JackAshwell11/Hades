"""Tests all functions in game_objects/components.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.exceptions import SpaceError
from hades.game_objects.components import Inventory

__all__ = ()


def test_inventory_init() -> None:
    """Test that an inventory is initialised correctly."""
    inventory = Inventory(3, 9)
    assert repr(inventory) == "<Inventory (Width=3) (Height=9)>"
    assert inventory.inventory == []


def test_inventory_add_item_to_inventory_valid() -> None:
    """Test that a valid item is added to the inventory correctly."""
    # Create the inventory and add an item
    inventory = Inventory(5, 6)
    inventory.add_item_to_inventory(1)

    # Check if the item was added correctly
    assert inventory.inventory == [1]


def test_inventory_add_item_to_inventory_zero_size() -> None:
    """Test that a valid item is not added to a zero size inventory."""
    # Create the zero-size inventory and add an item
    inventory = Inventory(0, 3)
    with pytest.raises(expected_exception=SpaceError):
        inventory.add_item_to_inventory("test")


def test_inventory_remove_item_from_inventory_valid() -> None:
    """Test that a valid item is removed from the inventory correctly."""
    # Create the inventory and add a few items
    inventory = Inventory(2, 4)
    inventory.add_item_to_inventory(1)
    inventory.add_item_to_inventory("test")
    inventory.add_item_to_inventory(3.14)

    # Check if an item can be removed correctly
    assert inventory.remove_item_from_inventory(1) == "test"
    assert inventory.inventory == [1, 3.14]


def test_inventory_remove_item_from_inventory_large_index() -> None:
    """Test that an exception is raised if a large index is provided."""
    # Create the inventory and add a few items
    inventory = Inventory(7, 2)
    inventory.add_item_to_inventory(5.9)
    inventory.add_item_to_inventory("temp")
    inventory.add_item_to_inventory(10)

    # Check if an exception is raised when an index larger than the inventory is
    # provided
    with pytest.raises(expected_exception=SpaceError):
        inventory.remove_item_from_inventory(10)
