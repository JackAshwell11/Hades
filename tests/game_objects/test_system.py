"""Tests all functions in game_objects/system.py."""
from __future__ import annotations

# Custom
from hades.game_objects.system import EntityComponentSystem

__all__ = ()


def test_ecs_init() -> None:
    """Test that the entity component system is initialised correctly."""
    entity_component_system = EntityComponentSystem()
    assert (
        repr(entity_component_system) == "<EntityComponentSystem (Game object count=0)>"
    )
    assert entity_component_system.game_objects == {}
    assert entity_component_system.processors == []


def test_ecs_add_game_object_zero_components() -> None:
    """Test that a game object is added to the system with zero components."""


def test_ecs_add_game_object_multiple_components() -> None:
    """Test that a game object is added to the system with one or more components."""


def test_ecs_remove_game_object_existing_id() -> None:
    """Test that a valid game object is removed from the system."""


def test_ecs_remove_game_object_non_existing_id() -> None:
    """Test that an invalid game object is removed from the system."""


def test_ecs_add_component_to_game_object_existing_id() -> None:
    """Test that a component is added to a valid game object."""


def test_ecs_add_component_to_game_object_non_existing_id() -> None:
    """Test that a component is not added to an invalid game object."""


def test_ecs_add_component_to_game_object_non_component() -> None:
    """Test that an invalid component is not added to a valid game object."""


def test_ecs_remove_component_from_game_object_existing_id() -> None:
    """Test that a valid component type is removed from a valid game object."""


def test_ecs_remove_component_from_game_object_non_existing_id() -> None:
    """Test that a valid component type is not removed from an invalid game object."""


def test_ecs_remove_component_from_game_object_non_existing_component() -> None:
    """Test that an invalid component type is not removed from a valid game object."""


def test_ecs_add_processor_valid_processor() -> None:
    """Test that a valid processor is added to the system."""


def test_ecs_add_processor_non_processor() -> None:
    """Test that an invalid processor is not added to the system."""


def test_ecs_remove_processor_existing_processor() -> None:
    """Test that an invalid processor is not added to the system."""


def test_ecs_remove_processor_non_existing_processor() -> None:
    """Test that an invalid processor is not added to the system."""


# TODO: REDO NAMES AND DOCSTRINGS OF THESE TESTS (BUT PLAN SHOULD BE GOOD)
