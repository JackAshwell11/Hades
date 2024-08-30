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
from hades_extensions.ecs import Registry

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ()


@pytest.fixture
def registry() -> Registry:
    """Get the registry for testing.

    Returns:
        Registry: The registry for testing.
    """
    registry = Registry()
    registry.add_systems()
    return registry


@pytest.fixture(scope="session")
def session_window() -> Generator[Window, None, None]:
    """Create a window for the session.

    Yields:
        The window for the session.
    """
    window = Window()
    yield window
    window.close()


@pytest.fixture
def window(session_window: Window) -> Window:
    """Create a window for testing.

    Args:
        session_window: The window for the session.

    Returns:
        The window for testing.
    """
    # Reset the window before running the test
    default_texture_cache.flush()
    session_window.switch_to()
    session_window.hide_view()
    session_window.dispatch_pending_events()
    session_window.flip()
    session_window.clear()
    session_window.default_camera.use()

    # Reset the context and run the garbage collector
    ctx = session_window.ctx
    ctx.gc_mode = "context_gc"
    ctx.reset()
    ctx.gc()
    gc.collect()
    return session_window
