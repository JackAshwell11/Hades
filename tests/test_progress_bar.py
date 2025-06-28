# pylint: disable=redefined-outer-name
"""Tests all classes and functions in progress_bar.py."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING
from unittest.mock import Mock

# Pip
import pytest
from arcade import color

# Custom
from hades.constructors import GameObjectConstructor, IconType
from hades.progress_bar import ProgressBar
from hades.sprite import HadesSprite
from hades_extensions.ecs import GameObjectType, Registry
from hades_extensions.ecs.components import Health

if TYPE_CHECKING:
    from arcade.types.color import RGBA255

__all__ = ()


@pytest.fixture
def health() -> Health:
    """Get a health stat for testing.

    Returns:
        A health stat for testing.
    """
    health = Mock(spec=Health)
    health.get_value.return_value = 100
    health.get_max_value.return_value = 100

    def set_value(value: int) -> None:
        health.get_value.return_value = max(0, min(value, health.get_max_value()))

    health.set_value.side_effect = set_value
    return health


@pytest.fixture
def sprite() -> HadesSprite:
    """Get a sprite for testing.

    Returns:
        A sprite for testing.
    """
    health = Mock(spec=Health)
    health.get_value.return_value = 100
    health.get_max_value.return_value = 100

    def set_value(value: int) -> None:
        health.get_value.return_value = max(0, min(value, health.get_max_value()))

    health.set_value.side_effect = set_value

    registry = Mock(spec=Registry)
    registry.get_component.return_value = health
    return HadesSprite(
        registry,
        0,
        (0, 0),
        GameObjectConstructor(
            "Test sprite",
            "Test description",
            GameObjectType.Player,
            0,
            [IconType.FLOOR],
            [
                (Health, (1, 1), color.GREEN),
            ],
        ),
    )


@pytest.mark.parametrize(
    ("init_data", "expected_width", "expected_height"),
    [
        (((0.5, 0.5), color.GREEN, 0), 29, 9),
        (((1, 1), color.YELLOW, 0), 58, 18),
        (((1, 1), color.YELLOW, 1), 58, 18),
        (((2, 2), color.GREEN, 0), 116, 36),
    ],
)
def test_progress_bar_init(
    sprite: HadesSprite,
    init_data: tuple[tuple[float, float], RGBA255, int],
    expected_width: int,
    expected_height: int,
) -> None:
    """Test that a progress bar initialises correctly.

    Args:
        sprite: A sprite for testing.
        init_data: The data to initialise the progress bar with.
        expected_width: The expected width of the progress bar
        expected_height: The expected height of the progress bar
    """
    progress_bar = ProgressBar((sprite, Health), *init_data)
    assert progress_bar.width == expected_width
    assert progress_bar.height == expected_height
    assert progress_bar.size_hint is None
    assert progress_bar.target_sprite == sprite
    assert (
        progress_bar.target_component
        == progress_bar.target_sprite.registry.get_component(
            sprite.game_object_id,
            Health,
        )
    )
    assert progress_bar.actual_bar.color == init_data[1]
    assert progress_bar.order == init_data[2]
    assert progress_bar.actual_bar in progress_bar.children


@pytest.mark.parametrize("scale", [(0, 0), (-1, -1)])
def test_progress_bar_init_invalid_scale(scale: tuple[float, float]) -> None:
    """Test that a progress bar with an invalid scale raises an error.

    Args:
        scale: The scale to initialise the progress bar with.
    """
    with pytest.raises(
        expected_exception=ValueError,
        match="Scale must be greater than 0",
    ):
        ProgressBar(Mock(), scale, color.GREEN, 0)


def test_progress_bar_init_invalid_order() -> None:
    """Test that a progress bar with an invalid order raises an error."""
    with pytest.raises(
        expected_exception=ValueError,
        match="Order must be greater than or equal to 0",
    ):
        ProgressBar(Mock(), (1, 1), color.GREEN, -1)


@pytest.mark.parametrize(
    ("new_value", "expected_size_hint", "expected_visibility"),
    [
        (-50, (0, 1), False),
        (0, (0, 1), False),
        (50, (0.5, 1), True),
        (100, (1, 1), True),
        (200, (1, 1), True),
    ],
)
def test_progress_bar_on_update(
    sprite: HadesSprite,
    new_value: int,
    expected_size_hint: tuple[float, float],
    *,
    expected_visibility: bool,
) -> None:
    """Test that a progress bar is correctly updated.

    Args:
        sprite: A sprite for testing.
        new_value: The new value of the health stat.
        expected_size_hint: The expected size hint of the actual bar.
        expected_visibility: The expected visibility of the actual bar.
    """
    progress_bar = ProgressBar((sprite, Health), (1, 1), color.GREEN, 0)
    progress_bar.target_component.set_value(new_value)
    progress_bar.on_update(0)
    assert progress_bar.actual_bar.size_hint == expected_size_hint
    assert progress_bar.actual_bar.visible == expected_visibility
