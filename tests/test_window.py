"""Tests all classes and functions in window.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Builtin
from unittest.mock import Mock

from arcade import get_default_texture
from arcade.gui import UIImage

# Pip
from PIL import Image

# Custom
from hades import SceneType
from hades.model import HadesModel
from hades.window import HadesWindow, main
from hades_extensions import GameEngine

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_hades_window_init(hades_window: HadesWindow) -> None:
    """Test that the hades window object initialises correctly.

    Args:
        hades_window: The hades window for testing.
    """
    assert isinstance(hades_window.model, HadesModel)
    assert hades_window.background_image.texture == get_default_texture()
    assert isinstance(hades_window.background_image, UIImage)
    assert len(hades_window.scenes) > 0


def test_hades_window_setup(hades_window: HadesWindow) -> None:
    """Test that the hades window object sets up correctly.

    Args:
        hades_window: The hades window for testing.
    """
    mock_center_window = Mock()
    mock_game_engine = Mock(spec=GameEngine)
    mock_show_view = Mock()
    hades_window.center_window = mock_center_window  # type: ignore[method-assign]
    hades_window.model._game_engine = mock_game_engine  # noqa: SLF001
    hades_window.show_view = mock_show_view  # type: ignore[method-assign]
    hades_window.setup()
    mock_center_window.assert_called_once()
    mock_game_engine.setup.assert_called_once()
    mock_show_view.assert_called_once_with(hades_window.scenes[SceneType.START_MENU])


def test_hades_window_save_background(
    hades_window: HadesWindow,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that the hades window saves the background correctly.

    Args:
        hades_window: The hades window for testing.
        monkeypatch: The monkeypatch fixture for patching functions.
    """
    mock_image = Mock(spec=Image.new("RGBA", (1, 1)))
    real_image = Image.new("RGBA", (10, 10))
    mock_image.filter.return_value = real_image
    monkeypatch.setattr("hades.window.get_image", lambda: mock_image)
    hades_window.save_background()
    assert getattr(hades_window.background_image.texture, "image", None) == real_image


def test_main_setup_run(monkeypatch: MonkeyPatch) -> None:
    """Test that the main function sets up and runs the window.

    Args:
        monkeypatch: The monkeypatch fixture for patching functions.
    """
    mock_window = Mock(spec=HadesWindow)
    monkeypatch.setattr("hades.window.HadesWindow", lambda: mock_window)
    main()
    mock_window.setup.assert_called_once()
    mock_window.run.assert_called_once()
