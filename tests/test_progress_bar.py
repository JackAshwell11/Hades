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
from hades.constructors import GameObjectConstructor
from hades.progress_bar import ProgressBar, ProgressBarGroup
from hades.sprite import HadesSprite
from hades_extensions.ecs import ComponentBase, GameObjectType, Registry
from hades_extensions.ecs.components import Armour, Health

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
def sprite(health: Health) -> HadesSprite:
    """Get a sprite for testing.

    Args:
        health: A health stat for testing.

    Returns:
        A sprite for testing.
    """
    registry = Mock(spec=Registry)
    registry.get_component.return_value = health
    return HadesSprite(
        registry,
        0,
        (0, 0),
        GameObjectConstructor(
            "Test",
            "Test description",
            GameObjectType.Player,
            [":resources:floor.png"],
            {
                Health: (0, 1, color.GREEN),
            },
        ),
    )


@pytest.mark.parametrize(
    ("init_data", "expected_width", "expected_height"),
    [
        ((0.5, color.GREEN), 27, 7),
        ((1, color.YELLOW), 54, 14),
        ((2, color.GREEN), 108, 28),
    ],
)
def test_progress_bar_init(
    health: Health,
    init_data: tuple[int, RGBA255],
    expected_width: int,
    expected_height: int,
) -> None:
    """Test that a ProgressBar is initialised correctly.

    Args:
        health: A health stat for testing.
        init_data: The data to initialise the ProgressBar with.
        expected_width: The expected width of the progress bar
        expected_height: The expected height of the progress bar
    """
    progress_bar = ProgressBar(health, *init_data)
    assert progress_bar.width == expected_width
    assert progress_bar.height == expected_height
    assert progress_bar.size_hint is None
    assert progress_bar.target_component == health
    assert progress_bar.actual_bar.color == init_data[1]
    assert progress_bar.actual_bar in progress_bar.children


@pytest.mark.parametrize("scale", [0, -1])
def test_progress_bar_init_negative_scale(scale: float) -> None:
    """Test that a ProgressBar with a negative scale raises an error.

    Args:
        scale: The scale to initialise the ProgressBar with.
    """
    with pytest.raises(
        expected_exception=ValueError,
        match="Scale must be greater than 0",
    ):
        ProgressBar(Mock(), scale, color.GREEN)


@pytest.mark.parametrize(
    "progress_bars",
    [
        ({Health: (0, 1, color.GREEN), Armour: (1, 1, color.YELLOW)}),
        ({Health: (1, 1, color.GREEN), Armour: (0, 1, color.YELLOW)}),
        ({Armour: (0, 1, color.YELLOW)}),
    ],
)
def test_progress_bar_group_init(
    sprite: HadesSprite,
    progress_bars: dict[type[ComponentBase], tuple[int, float, RGBA255]],
) -> None:
    """Test that a ProgressBarGroup is initialised correctly.

    Args:
        sprite: A sprite for testing.
        progress_bars: The game object's progress bars.
    """
    # Set up the mock components
    mock_health = Mock(spec=Health)
    mock_health.get_value.return_value = 100
    mock_health.get_max_value.return_value = 100
    mock_armour = Mock(spec=Armour)
    mock_armour.get_value.return_value = 50
    mock_armour.get_max_value.return_value = 50

    sprite.constructor.progress_bars = progress_bars
    progress_bar_group = ProgressBarGroup(sprite)
    assert len(progress_bar_group.children) == len(progress_bars)
    for component in (mock_health, mock_armour):
        if type(component) in progress_bars:
            index = progress_bars[type(component)][0]
            assert progress_bar_group.children[index].target_component == component
        else:
            assert not any(
                progress_bar.target_component == component
                for progress_bar in progress_bar_group.children
            )


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
def test_progress_bar_group_on_update(
    sprite: HadesSprite,
    new_value: int,
    expected_size_hint: tuple[float, float],
    *,
    expected_visibility: bool,
) -> None:
    """Test that a ProgressBarGroup is correctly updated.

    Args:
        sprite: A sprite for testing.
        new_value: The new value of the health stat.
        expected_size_hint: The expected size hint of the actual bar.
        expected_visibility: The expected visibility of the actual bar.
    """
    progress_bar_group = ProgressBarGroup(sprite)
    progress_bar_group.children[0].target_component.set_value(new_value)
    progress_bar_group.on_update(0)
    assert progress_bar_group.children[0].actual_bar.size_hint == expected_size_hint
    assert progress_bar_group.children[0].actual_bar.visible == expected_visibility
