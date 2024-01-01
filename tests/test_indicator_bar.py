"""Tests all classes and functions in indicator_bar.py."""

from __future__ import annotations

# Pip
import pytest
from arcade import SpriteList, SpriteSolidColor, color

# Custom
from hades.constants import GameObjectType
from hades.indicator_bar import IndicatorBar, IndicatorBarError
from hades.sprite import HadesSprite
from hades_extensions.game_objects.components import Stat

__all__ = ()


@pytest.fixture()
def sprite() -> HadesSprite:
    """Get a sprite for testing.

    Returns:
        A sprite for testing.
    """
    return HadesSprite((0, GameObjectType.PLAYER), (0, 0), ["floor.png"])


@pytest.fixture()
def stat() -> Stat:
    """Get a stat for testing.

    Returns:
        A stat for testing.
    """
    return Stat(100, -1)


@pytest.fixture()
def spritelist() -> SpriteList[SpriteSolidColor]:
    """Get a sprite list for testing.

    Returns:
        A sprite list for testing.
    """
    return SpriteList[SpriteSolidColor]()


def test_raise_indicator_bar_error() -> None:
    """Test that IndicatorBarError is raised correctly."""
    with pytest.raises(
        expected_exception=IndicatorBarError,
        match="Index must be greater than or equal to 0.",
    ):
        raise IndicatorBarError


def test_indicator_bar_init(
    sprite: HadesSprite,
    stat: Stat,
    spritelist: SpriteList[SpriteSolidColor],
) -> None:
    """Test that an IndicatorBar is initialised correctly.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        spritelist: A sprite list for testing.
    """
    indicator_bar = IndicatorBar(sprite, stat, spritelist)
    assert indicator_bar.background_box.width == 54
    assert indicator_bar.background_box.height == 14
    assert indicator_bar.background_box.position == (0, 0)
    assert indicator_bar.background_box.color == color.BLACK
    assert indicator_bar.actual_bar.width == 50
    assert indicator_bar.actual_bar.height == 10
    assert indicator_bar.actual_bar.position == (0, 0)
    assert indicator_bar.actual_bar.color == color.RED
    assert indicator_bar.offset == 0


def test_indicator_bar_positive_index(
    sprite: HadesSprite,
    stat: Stat,
    spritelist: SpriteList[SpriteSolidColor],
) -> None:
    """Test that an IndicatorBar is initialised correctly with a positive index.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        spritelist: A sprite list for testing.
    """
    assert IndicatorBar(sprite, stat, spritelist, 1).offset == 14


def test_indicator_bar_negative_index(
    sprite: HadesSprite,
    stat: Stat,
    spritelist: SpriteList[SpriteSolidColor],
) -> None:
    """Test that an IndicatorBar with a negative index raises an error.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        spritelist: A sprite list for testing.
    """
    with pytest.raises(
        expected_exception=IndicatorBarError,
        match="Index must be greater than or equal to 0.",
    ):
        IndicatorBar(sprite, stat, spritelist, -1)


def test_indicator_bar_on_update_no_position(
    sprite: HadesSprite,
    stat: Stat,
    spritelist: SpriteList[SpriteSolidColor],
) -> None:
    """Test that an IndicatorBar is updated if the target sprite hasn't moved.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        spritelist: A sprite list for testing.
    """
    indicator_bar = IndicatorBar(sprite, stat, spritelist)
    indicator_bar.on_update(0)
    assert indicator_bar.background_box.position == (32, 96)
    assert indicator_bar.actual_bar.position == (32, 96)


def test_indicator_bar_on_update_position(
    sprite: HadesSprite,
    stat: Stat,
    spritelist: SpriteList[SpriteSolidColor],
) -> None:
    """Test that an IndicatorBar is updated if the target sprite has moved.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        spritelist: A sprite list for testing.
    """
    indicator_bar = IndicatorBar(sprite, stat, spritelist)
    sprite.center_x += 100
    sprite.center_y += 50
    indicator_bar.on_update(0)
    assert indicator_bar.background_box.position == (132, 146)
    assert indicator_bar.actual_bar.position == (132, 146)


def test_indicator_bar_on_update_position_offset(
    sprite: HadesSprite,
    stat: Stat,
    spritelist: SpriteList[SpriteSolidColor],
) -> None:
    """Test that an offset IndicatorBar is updated if the target sprite has moved.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        spritelist: A sprite list for testing.
    """
    indicator_bar = IndicatorBar(sprite, stat, spritelist, 1)
    sprite.center_x += 100
    sprite.center_y += 50
    indicator_bar.on_update(0)
    assert indicator_bar.background_box.position == (132, 160)
    assert indicator_bar.actual_bar.position == (132, 160)


def test_indicator_bar_on_update_value_not_changed(
    sprite: HadesSprite,
    stat: Stat,
    spritelist: SpriteList[SpriteSolidColor],
) -> None:
    """Test that an IndicatorBar is updated if the target value hasn't changed.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        spritelist: A sprite list for testing.
    """
    indicator_bar = IndicatorBar(sprite, stat, spritelist)
    indicator_bar.on_update(0)
    assert indicator_bar.actual_bar.width == 50


def test_indicator_bar_on_update_value_changed(
    sprite: HadesSprite,
    stat: Stat,
    spritelist: SpriteList[SpriteSolidColor],
) -> None:
    """Test that an IndicatorBar is updated if the target value has changed.

    Args:
        sprite: A sprite for testing.
        stat: A stat for testing.
        spritelist: A sprite list for testing.
    """
    indicator_bar = IndicatorBar(sprite, stat, spritelist)
    stat.set_value(60)
    indicator_bar.on_update(0)
    assert indicator_bar.actual_bar.width == 30
