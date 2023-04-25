"""Tests all functions in game_objects/attributes.py."""
from __future__ import annotations

# Pip
import pytest

# Custom
from hades.game_objects.attributes import (
    EntityAttributeError,
)

__all__ = ()


def test_raise_entity_attribute_error() -> None:
    """Test that EntityAttributeError is raised correctly."""
    with pytest.raises(
        expected_exception=EntityAttributeError,
        match="The entity attribute `test` cannot be set.",
    ):
        raise EntityAttributeError(name="test", error="be set")


# TODO: TEST WITH HEALTH, SPEED, AND MONEY
