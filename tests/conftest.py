"""Holds common fixtures for the tests."""

from __future__ import annotations

# Pip
import pytest

# Custom
from hades_extensions.game_objects import Registry

__all__ = ()


@pytest.fixture()
def registry() -> Registry:
    """Get the registry for testing.

    Returns:
        Registry: The registry for testing.
    """
    return Registry()
