# pylint: disable=redefined-outer-name
"""Holds common fixtures for the tests."""

from __future__ import annotations

# Builtin
import gc
from typing import TYPE_CHECKING

# Pip
import pytest
from arcade import Window
from arcade.texture import default_texture_cache

# Custom
from hades_extensions.game_objects import Registry

if TYPE_CHECKING:
    from typing import Final

__all__ = ()


# Create a global window for testing. This should be reset before each test
WINDOW: Final[Window] = Window()


@pytest.fixture
def registry() -> Registry:
    """Get the registry for testing.

    Returns:
        Registry: The registry for testing.
    """
    registry = Registry()
    registry.add_systems()
    return registry


@pytest.fixture
def window() -> Window:
    """Create a window for testing.

    Returns:
        The window for testing.
    """
    # Reset the window before running the test
    default_texture_cache.flush()
    WINDOW.switch_to()
    WINDOW.hide_view()
    WINDOW.dispatch_pending_events()
    WINDOW.flip()
    WINDOW.clear()
    WINDOW.default_camera.use()

    # Reset the context and run the garbage collector
    ctx = WINDOW.ctx
    ctx.gc_mode = "context_gc"
    ctx.reset()
    ctx.gc()
    gc.collect()
    return WINDOW
