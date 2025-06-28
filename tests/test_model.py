"""Tests all classes and functions in model.py."""

from __future__ import annotations

# Builtin
from unittest.mock import Mock

# Custom
from hades.model import HadesModel
from hades_extensions import GameEngine
from hades_extensions.ecs import Registry


def test_hades_model_init() -> None:
    """Test that the hades model object initialises correctly."""
    model = HadesModel()
    assert isinstance(model.game_engine, GameEngine)


def test_hades_model_registry() -> None:
    """Test that the registry property returns the correct value."""
    model = HadesModel()
    mock_registry = Mock(spec=Registry)
    mock_game_engine = Mock(spec=GameEngine)
    mock_game_engine.registry = mock_registry
    model.game_engine = mock_game_engine
    assert model.registry == mock_registry


def test_hades_model_player_id() -> None:
    """Test that the player_id property returns the correct value."""
    model = HadesModel()
    mock_game_engine = Mock(spec=GameEngine)
    mock_game_engine.player_id = 42
    model.game_engine = mock_game_engine
    assert model.player_id == 42
