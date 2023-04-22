"""Tests all functions in game_objects/components.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.exceptions import SpaceError
from hades.game_objects.components import Graphics, Inventory
from hades.textures import TextureType

__all__ = ()


def test_graphics_init_zero_textures() -> None:
    """Test that graphics is initialised correctly with zero textures."""
    graphics = Graphics(set())
    assert repr(graphics) == "<Graphics (Texture count=0) (Blocking=False)>"
    assert graphics.textures == {}


def test_graphics_init_multiple_textures() -> None:
    """Test that graphics is initialised correctly with multiple textures."""
    graphics = Graphics({TextureType.WALL, TextureType.FLOOR}, blocking=True)
    assert repr(graphics) == "<Graphics (Texture count=2) (Blocking=True)>"
    assert graphics.textures == {
        TextureType.WALL.name: TextureType.WALL.value,
        TextureType.FLOOR.name: TextureType.FLOOR.value,
    }


def test_inventory_init() -> None:
    """Test that an inventory is initialised correctly."""
    inventory = Inventory(3, 9)
    assert repr(inventory) == "<Inventory (Width=3) (Height=9)>"
    assert not inventory.inventory


def test_inventory_add_item_to_inventory_valid() -> None:
    """Test that a valid item is added to the inventory correctly."""
    inventory = Inventory(5, 6)
    inventory.add_item_to_inventory(1)
    assert inventory.inventory == [1]


def test_inventory_add_item_to_inventory_zero_size() -> None:
    """Test that a valid item is not added to a zero size inventory."""
    inventory = Inventory(0, 3)
    with pytest.raises(expected_exception=SpaceError):
        inventory.add_item_to_inventory("test")


def test_inventory_remove_item_from_inventory_valid() -> None:
    """Test that a valid item is removed from the inventory correctly."""
    inventory = Inventory(2, 4)
    inventory.add_item_to_inventory(1)
    inventory.add_item_to_inventory("test")
    inventory.add_item_to_inventory(3.14)
    assert inventory.remove_item_from_inventory(1) == "test"
    assert inventory.inventory == [1, 3.14]


def test_inventory_remove_item_from_inventory_large_index() -> None:
    """Test that an exception is raised if a large index is provided."""
    inventory = Inventory(7, 2)
    inventory.add_item_to_inventory(5.9)
    inventory.add_item_to_inventory("temp")
    inventory.add_item_to_inventory(10)
    with pytest.raises(expected_exception=SpaceError):
        inventory.remove_item_from_inventory(10)
