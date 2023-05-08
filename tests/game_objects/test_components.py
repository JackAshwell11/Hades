"""Tests all functions in game_objects/components.py."""
from __future__ import annotations

# Pip
import arcade
import pytest

# Custom
from hades.game_objects.components import (
    Graphics,
    InstantEffects,
    Inventory,
    InventorySpaceError,
    StatusEffects,
)
from hades.textures import TextureType

__all__ = ()


def test_raise_inventory_space_error() -> None:
    """Test that SpaceError is raised correctly."""
    with pytest.raises(
        expected_exception=InventorySpaceError,
        match="The inventory is full.",
    ):
        raise InventorySpaceError(full=True)


def test_graphics_init_zero_textures() -> None:
    """Test that graphics is initialised correctly with zero textures."""
    graphics = Graphics(set())
    assert repr(graphics) == "<Graphics (Texture count=0) (Blocking=False)>"
    assert graphics.textures_dict == {}


def test_graphics_init_multiple_textures() -> None:
    """Test that graphics is initialised correctly with multiple textures."""
    graphics = Graphics({TextureType.WALL, TextureType.FLOOR}, blocking=True)
    assert repr(graphics) == "<Graphics (Texture count=2) (Blocking=True)>"
    assert graphics.textures_dict == {
        TextureType.WALL: TextureType.WALL.value,
        TextureType.FLOOR: TextureType.FLOOR.value,
    }


def test_graphics_add_to_spritelist_not_already_added() -> None:
    """Test that adding the graphics component to a spritelist is successful."""
    spritelist = arcade.SpriteList()
    graphics = Graphics({TextureType.FLOOR}).add_to_spritelist(spritelist)
    assert spritelist.sprite_list == [graphics]


def test_graphics_add_to_spritelist_already_added() -> None:
    """Test that the same graphics component twice to a spritelist raises an error."""
    spritelist = arcade.SpriteList()
    with pytest.raises(expected_exception=ValueError):
        Graphics({TextureType.FLOOR}).add_to_spritelist(spritelist).add_to_spritelist(
            spritelist,
        )


def test_instant_effect_init() -> None:
    """Test that instant effect is initialised correctly."""
    assert (
        repr(InstantEffects(lambda level: 2**level, 10))
        == "<InstantEffect (Level limit=10)>"
    )


def test_inventory_init() -> None:
    """Test that an inventory is initialised correctly."""
    inventory = Inventory[int](3, 9)
    assert repr(inventory) == "<Inventory (Width=3) (Height=9)>"
    assert not inventory.inventory


def test_inventory_add_item_to_inventory_valid() -> None:
    """Test that a valid item is added to the inventory correctly."""
    inventory = Inventory[str](5, 6)
    inventory.add_item_to_inventory("test")
    assert inventory.inventory == ["test"]


def test_inventory_add_item_to_inventory_zero_size() -> None:
    """Test that a valid item is not added to a zero size inventory."""
    inventory = Inventory[float](0, 3)
    with pytest.raises(expected_exception=InventorySpaceError):
        inventory.add_item_to_inventory(5.0)


def test_inventory_add_item_to_inventory_multiple_types() -> None:
    """Test that multiple types cannot be given to the inventory constructor."""
    with pytest.raises(expected_exception=TypeError):
        Inventory[int, str](4, 6)  # type: ignore[misc]


def test_inventory_remove_item_from_inventory_valid() -> None:
    """Test that a valid item is removed from the inventory correctly."""
    inventory = Inventory[int](2, 4)
    inventory.add_item_to_inventory(1)
    inventory.add_item_to_inventory(7)
    inventory.add_item_to_inventory(4)
    assert inventory.remove_item_from_inventory(1) == 7
    assert inventory.inventory == [1, 4]


def test_inventory_remove_item_from_inventory_large_index() -> None:
    """Test that an exception is raised if a large index is provided."""
    inventory = Inventory[str](7, 2)
    inventory.add_item_to_inventory("test")
    inventory.add_item_to_inventory("temp")
    inventory.add_item_to_inventory("inventory")
    with pytest.raises(expected_exception=InventorySpaceError):
        inventory.remove_item_from_inventory(10)


def test_status_effect_init() -> None:
    """Test that status effect is initialised correctly."""
    assert (
        repr(StatusEffects(lambda level: 2**level, lambda level: 3 * level + 5, 5))
        == "<StatusEffect (Level limit=5)>"
    )
