# pylint: disable=redefined-outer-name
"""Tests all classes and functions in indicator_bar.py."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING
from unittest.mock import Mock

# Pip
import pytest
from arcade import color

from hades.constructors import GameObjectConstructor
from hades.indicator_bar import IndicatorBar, IndicatorBarError

# Custom
from hades.sprite import HadesSprite
from hades_extensions.game_objects import GameObjectType, Vec2d
from hades_extensions.game_objects.components import Stat

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

__all__ = ()


@pytest.fixture
def sprite() -> HadesSprite:
    """Get a sprite for testing.

    Returns:
        A sprite for testing.
    """
    return HadesSprite(
        Mock(),
        0,
        Vec2d(0, 0),
        GameObjectConstructor(
            "Test",
            "Test description",
            GameObjectType.Player,
            [":resources:floor.png"],
            [],
        ),
    )


@pytest.fixture
def stat() -> Stat:
    """Get a stat for testing.

    Returns:
        A stat for testing.
    """
    return Stat(100, -1)


def test_raise_indicator_bar_error() -> None:
    """Test that IndicatorBarError is raised correctly."""
    with pytest.raises(
        expected_exception=IndicatorBarError,
        match="Index must be greater than or equal to 0.",
    ):
        raise IndicatorBarError


@pytest.mark.parametrize(
    ("index", "expected_offset"),
    [
        (0, 0),
        (1, 14),
        (2, 28),
    ],
)
def test_indicator_bar_init(
    index: int,
    expected_offset: int,
    monkeypatch: MonkeyPatch,
    sprite: HadesSprite,
    stat: Stat,
) -> None:
    """Test that an IndicatorBar is initialised correctly.

    Args:
        index: The index of the IndicatorBar.
        expected_offset: The expected offset of the IndicatorBar.
        monkeypatch: A monkeypatch for testing.
        sprite: A sprite for testing.
        stat: A stat for testing.
    """
    # Set up the required monkeypatches
    mock_spritelist = Mock()
    mock_append = Mock()
    monkeypatch.setattr(mock_spritelist, "append", mock_append)

    # Make sure the IndicatorBar is initialised correctly
    indicator_bar = IndicatorBar(sprite, stat, mock_spritelist, index)
    assert indicator_bar.background_box.width == 54
    assert indicator_bar.background_box.height == 14
    assert indicator_bar.background_box.position == (0, 0)
    assert indicator_bar.background_box.color == color.BLACK
    assert indicator_bar.actual_bar.width == 50
    assert indicator_bar.actual_bar.height == 10
    assert indicator_bar.actual_bar.position == (0, 0)
    assert indicator_bar.actual_bar.color == color.RED
    assert indicator_bar.offset == expected_offset
    assert mock_append.call_count == 2


def test_indicator_bar_negative_index() -> None:
    """Test that an IndicatorBar with a negative index raises an error."""
    with pytest.raises(
        expected_exception=IndicatorBarError,
        match="Index must be greater than or equal to 0.",
    ):
        IndicatorBar(Mock(), Mock(), Mock(), -1)


@pytest.mark.parametrize(
    (
        "index",
        "centre_diff",
        "expected_position",
    ),
    [
        (0, (0, 0), (32, 80)),
        (0, (100, 50), (132, 130)),
        (1, (0, 0), (32, 94)),
        (1, (100, 50), (132, 144)),
    ],
)
def test_indicator_bar_on_update_position(
    index: int,
    centre_diff: tuple[int, int],
    expected_position: tuple[int, int],
    sprite: HadesSprite,
    stat: Stat,
) -> None:
    """Test that an IndicatorBar's position is updated correctly.

    Args:
        index: The index of the IndicatorBar.
        centre_diff: The difference in the centre of the sprite.
        expected_position: The expected position of the IndicatorBar.
        sprite: A sprite for testing.
        stat: A stat for testing.
    """
    indicator_bar = IndicatorBar(sprite, stat, Mock(), index)
    sprite.center_x += centre_diff[0]
    sprite.center_y += centre_diff[1]
    indicator_bar.on_update(0)
    assert indicator_bar.background_box.position == expected_position
    assert indicator_bar.actual_bar.position == expected_position


@pytest.mark.parametrize(
    ("new_value", "expected_width"),
    [
        (0, 0),
        (50, 25),
        (200, 50),
        (100, 50),
        (-50, 0),
    ],
)
def test_indicator_bar_on_update_value_changed(
    new_value: int,
    expected_width: int,
    sprite: HadesSprite,
    stat: Stat,
) -> None:
    """Test that an IndicatorBar is updated if the target value has changed.

    Args:
        new_value: The new value of the stat.
        expected_width: The expected width of the actual bar.
        sprite: A sprite for testing.
        stat: A stat for testing.
    """
    indicator_bar = IndicatorBar(sprite, stat, Mock())
    stat.set_value(new_value)
    indicator_bar.on_update(0)
    assert indicator_bar.actual_bar.width == expected_width
