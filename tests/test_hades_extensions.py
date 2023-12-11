"""Tests all classes and functions in the hades_extensions module."""

from __future__ import annotations

# Pip
import pytest

# Custom
from hades_extensions.game_objects import (
    AttackAlgorithm,
    Registry,
)
from hades_extensions.game_objects.components import (
    Armour,
    ArmourRegen,
    Attacks,
    EffectApplier,
    Health,
    Inventory,
    StatusEffects,
    Upgrades,
)
from hades_extensions.generation import create_map


@pytest.fixture()
def registry() -> Registry:
    """Create a registry for testing.

    Returns:
        The registry object.
    """
    registry = Registry()
    registry.add_systems()
    return registry


# TODO: these two fixtures should have every component combined


@pytest.fixture()
def game_object_one(registry: Registry) -> int:
    """Create a game object for testing.

    Args:
        registry: The registry object.

    Returns:
        The game object id.
    """
    # TODO: this should have every component for a player
    return registry.create_game_object([
        Health(100, -1),
        Armour(50, -1),
        ArmourRegen(2, -1),
        Upgrades(),
        Attacks([AttackAlgorithm.Ranged]),
        Inventory(10, 10),
        StatusEffects(),
    ])


@pytest.fixture()
def game_object_two(registry: Registry) -> int:
    """Create a game object for testing.

    Args:
        registry: The registry object.

    Returns:
        The game object id.
    """
    # TODO: this should have every component for an enemy
    return registry.create_game_object([
        Health(75, -1),
        Armour(25, -1),
        EffectApplier(),
    ])


def test_game_objects_registry() -> None:
    """Test that the registry works correctly."""


def test_game_objects_systems() -> None:
    """Test that the systems work correctly."""


# def test_armour_regen_system(registry: Registry) -> None:
#     """Test that the armour regen system works correctly."""
#     armour = Armour(100, -1)
#     armour.set_value(50)
#     registry.create_game_object([armour, ArmourRegen(2, -1)])
#     registry.get_system(ArmourRegenSystem).update(2)
#     registry.get_system(ArmourRegenSystem).update(2)
#     registry.get_system(ArmourRegenSystem).update(1)
#     assert armour.get_value() == 52
#
#
# def test_attack_system(registry: Registry) -> None:
#     """Test that the attack system works correctly."""
#     registry.create_game_object([Attacks([AttackAlgorithm.Ranged])], kinematic=True)
#     assert registry.get_system(AttackSystem).do_attack(0, []) == (Vec2d(0, 0), 300, 0)
#
#
# def test_damage_system(registry: Registry) -> None:
#     """Test that the damage system works correctly."""
#     registry.create_game_object([Health(100, -1), Armour(100, -1)])
#     registry.get_system(DamageSystem).deal_damage(0, 50)
#     assert registry.get_component(0, Health).get_value() == 100
#     assert registry.get_component(0, Armour).get_value() == 50
#
#
# def test_effect_system(registry: Registry) -> None:
#     """Test that the effect system works correctly."""
#
#
# def test_inventory_system(registry: Registry) -> None:
#     """Test that the inventory system works correctly."""
#     registry.create_game_object([Inventory(10, 10)])
#     with pytest.raises(expected_exception=InventorySpaceError, match="The inventory is empty."):
#         registry.get_system(InventorySystem).remove_item_from_inventory(0, 0)
#     registry.get_system(InventorySystem).add_item_to_inventory(0, 0)
#
#
# def test_footprint_system(registry: Registry) -> None:
#     """Test that the footprint system works correctly."""
#
#
# def test_keyboard_movement_system(registry: Registry) -> None:
#     """Test that the keyboard movement system works correctly."""
#
#
# def test_steering_movement_system(registry: Registry) -> None:
#     """Test that the steering movement system works correctly."""
#
#
# def test_upgrade_system(registry: Registry) -> None:
#     """Test that the upgrade system works correctly."""
#     registry.create_game_object([Health(100, 1), Upgrades()
#     registry.get_system(AttackSystem).upgrade(0, 0)
#     assert registry.get_component(0, Attacks).get_value() == [AttackAlgorithm.Ranged, AttackAlgorithm.Ranged]


def test_generation() -> None:
    """Test that the generation module works correctly."""
    assert create_map(0, 1)[1] == (0, 30, 20)
    assert create_map(1)[1] == (1, 36, 24)
