# pylint: disable=redefined-outer-name
"""Holds common fixtures for the tests."""

from __future__ import annotations

# Builtin
import gc
from typing import TYPE_CHECKING

# Pip
import pytest
from arcade import get_default_texture
from arcade.texture import default_texture_cache

# Custom
from hades.window import HadesWindow

if TYPE_CHECKING:
    from collections.abc import Generator

    from _pytest.monkeypatch import MonkeyPatch

__all__ = ()


@pytest.fixture(scope="session")
def session_window() -> Generator[HadesWindow]:
    """Create a window for the session.

    Yields:
        The hades window for the session.
    """
    hades_window = HadesWindow()
    yield hades_window
    hades_window.close()


@pytest.fixture
def hades_window(session_window: HadesWindow) -> HadesWindow:
    """Create a hades window for testing.

    Args:
        session_window: The hades window for the session.

    Returns:
        The hades window for testing.
    """
    # Reset the window before running the test
    default_texture_cache.flush()
    session_window.switch_to()
    session_window.hide_view()
    session_window.dispatch_pending_events()
    session_window.flip()
    session_window.clear()
    session_window.default_camera.use()
    session_window.background_image.texture = get_default_texture()

    # Reset the context and run the garbage collector
    ctx = session_window.ctx
    ctx.gc_mode = "context_gc"
    ctx.reset()
    ctx.gc()
    gc.collect()
    return session_window


@pytest.fixture
def sized_window(
    monkeypatch: MonkeyPatch,
    request: pytest.FixtureRequest,
    hades_window: HadesWindow,
) -> HadesWindow:
    """Create a window with a custom size for testing.

    Args:
        monkeypatch: The monkeypatch fixture for mocking.
        request: The fixture request object.
        hades_window: The hades window for testing.

    Returns:
        A window with a tom size for testing.
    """
    # We need to set these attributes since `set_size()` doesn't work in
    # headless mode
    monkeypatch.setattr(hades_window, "_width", request.param[0])
    monkeypatch.setattr(hades_window, "_height", request.param[1])
    return hades_window
