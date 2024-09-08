# pylint: disable=redefined-outer-name
"""Tests all classes and functions in indicator_bar.py."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING
from unittest.mock import Mock

# Pip
import pytest
from arcade import color

# Custom
from hades.constructors import GameObjectConstructor
from hades.indicator_bar import IndicatorBar
from hades.sprite import HadesSprite
from hades_extensions.ecs import GameObjectType, Vec2d
from hades_extensions.ecs.components import Stat

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from arcade import Window

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
        ),
    )


@pytest.fixture
def stat(monkeypatch: MonkeyPatch) -> Stat:
    """Get a stat for testing.

    Args:
        monkeypatch: A monkeypatch for testing.

    Returns:
        A stat for testing.
    """
    stat = Stat(100, -1)
    monkeypatch.setattr(
        "hades.indicator_bar.INDICATOR_BAR_COLOURS",
        {Stat: color.GREEN},
    )
    return stat


@pytest.mark.parametrize(
    ("index", "expected_offset"),
    [
        (0, 0),
        (1, 14),
        (2, 28),
    ],
)
def test_indicator_bar_init(
    monkeypatch: MonkeyPatch,
    sprite: HadesSprite,
    stat: Stat,
    index: int,
    expected_offset: int,
) -> None:
    """Test that an IndicatorBar is initialised correctly.

    Args:
        monkeypatch: The monkeypatch fixture for mocking.
        sprite: A sprite for testing.
        stat: A stat for testing.
        index: The index of the IndicatorBar.
        expected_offset: The expected offset of the IndicatorBar.
    """
    # Set up the required monkeypatches
    mock_spritelist = Mock()
    mock_append = Mock()
    monkeypatch.setattr(mock_spritelist, "append", mock_append)

    # Make sure the IndicatorBar is initialised correctly
    indicator_bar = IndicatorBar(
        (sprite, stat),
        mock_spritelist,
        index,
        fixed_position=False,
    )
    assert indicator_bar.background_box.width == 54
    assert indicator_bar.background_box.height == 14
    assert indicator_bar.background_box.position == (0, 0)
    assert indicator_bar.background_box.color == color.BLACK
    assert indicator_bar.actual_bar.width == 50
    assert indicator_bar.actual_bar.height == 10
    assert indicator_bar.actual_bar.position == (0, 0)
    assert indicator_bar.actual_bar.color == color.GREEN
    assert indicator_bar.offset == expected_offset
    assert mock_append.call_count == 2


@pytest.mark.parametrize(
    ("window_height", "expected_y"),
    [
        (140, 131),
        (480, 471),
        (720, 711),
        (1080, 1071),
        (144, 135),
        (2160, 2151),
    ],
)
def test_indicator_bar_init_fixed_position(
    monkeypatch: MonkeyPatch,
    window: Window,
    stat: Stat,
    window_height: int,
    expected_y: int,
) -> None:
    """Test that an IndicatorBar is initialised with a fixed position.

    Args:
        monkeypatch: The monkeypatch fixture for mocking.
        window: The window for testing.
        stat: A stat for testing.
        window_height: The height of the window.
        expected_y: The expected y position of the IndicatorBar.
    """
    # We need to set this attribute since `set_size()` doesn't work in headless
    # mode
    monkeypatch.setattr(window, "height", window_height)

    # Make sure the IndicatorBar is initialised correctly
    indicator_bar = IndicatorBar((Mock(), stat), Mock(), fixed_position=True)
    assert (
        indicator_bar.background_box.position
        == indicator_bar.actual_bar.position
        == (29, expected_y)
    )


# TODO: TEST UPDATING POSITION AND VALUE FOR FIXED POS


def test_indicator_bar_negative_index() -> None:
    """Test that an IndicatorBar with a negative index raises an error."""
    with pytest.raises(
        expected_exception=ValueError,
        match="Index must be greater than or equal to 0.",
    ):
        IndicatorBar((Mock(), Mock()), Mock(), -1, fixed_position=False)


def test_indicator_bar_invalid_component_colour() -> None:
    """Test that an IndicatorBar with an invalid component raises an error."""
    with pytest.raises(
        expected_exception=KeyError,
        match="<class 'unittest.mock.Mock'>",
    ):
        IndicatorBar((Mock(), Mock()), Mock(), fixed_position=False)


@pytest.mark.parametrize(
    (
        "init_data",
        "centre_diff",
        "expected_position",
    ),
    [
        ((0, False), (0, 0), (32, 80)),
        ((0, False), (100, 50), (132, 130)),
        ((1, False), (0, 0), (32, 94)),
        ((1, False), (100, 50), (132, 144)),
    ],
)
def test_indicator_bar_on_update_position(
    sprite: HadesSprite,
    stat: Stat,
    init_data: tuple[int, bool],
    centre_diff: tuple[int, int],
    expected_position: tuple[int, int],
) -> None:
    """Test that an IndicatorBar's position is updated correctly.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        init_data: The data to initialise the IndicatorBar with.
        centre_diff: The difference in the centre of the sprite.
        expected_position: The expected position of the IndicatorBar.
    """
    indicator_bar = IndicatorBar(
        (sprite, stat),
        Mock(),
        init_data[0],
        fixed_position=init_data[1],
    )
    sprite.center_x += centre_diff[0]
    sprite.center_y += centre_diff[1]
    indicator_bar.update()
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
    sprite: HadesSprite,
    stat: Stat,
    new_value: int,
    expected_width: int,
) -> None:
    """Test that an IndicatorBar is updated if the target value has changed.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        new_value: The new value of the stat.
        expected_width: The expected width of the actual bar.
    """
    indicator_bar = IndicatorBar((sprite, stat), Mock(), fixed_position=False)
    stat.set_value(new_value)
    indicator_bar.update()
    assert indicator_bar.actual_bar.width == expected_width
