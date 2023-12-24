"""Manages creation and updates of entity indicator bars."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import SpriteSolidColor, color

# Custom
from hades.constants import (
    INDICATOR_BAR_BORDER_SIZE,
    INDICATOR_BAR_DISTANCE,
    INDICATOR_BAR_HEIGHT,
    INDICATOR_BAR_WIDTH,
)

if TYPE_CHECKING:
    from arcade import SpriteList

    from hades.sprite import HadesSprite
    from hades_extensions.game_objects.components import Stat

__all__ = ("IndicatorBar", "IndicatorBarError")


class IndicatorBarError(Exception):
    """Raised when a value is less than a required value."""

    def __init__(self: IndicatorBarError) -> None:
        """Initialise the object."""
        super().__init__("Index must be greater than or equal to 0.")


class IndicatorBar:
    """Represents a variable bar that can display information about a game object.

    Attributes:
        background_box: The background box for the indicator bar.
        actual_bar: The actual bar of the indicator bar.
        offset: The offset of the indicator bar relative to the other indicator bars.
    """

    __slots__ = (
        "target_sprite",
        "target_component",
        "background_box",
        "actual_bar",
        "offset",
    )

    def __init__(
        self: IndicatorBar,
        target_sprite: HadesSprite,
        target_component: Stat,
        spritelist: SpriteList[SpriteSolidColor],
        index: int = 0,
    ) -> None:
        """Initialise the object.

        Args:
            target_sprite: The sprite that the indicator bar will be attached to.
            target_component: The component to get the information from.
            spritelist: The spritelist to add the indicator bar sprites to.
            index: The index of the indicator bar relative to the other indicator bars.

        Raises:
            IndicatorBarError: If the index is less than 0.
        """
        if index < 0:
            raise IndicatorBarError
        self.target_sprite: HadesSprite = target_sprite
        self.target_component: Stat = target_component
        self.background_box: SpriteSolidColor = SpriteSolidColor(
            INDICATOR_BAR_WIDTH + INDICATOR_BAR_BORDER_SIZE,
            INDICATOR_BAR_HEIGHT + INDICATOR_BAR_BORDER_SIZE,
            color=color.BLACK,
        )
        self.actual_bar: SpriteSolidColor = SpriteSolidColor(
            INDICATOR_BAR_WIDTH,
            INDICATOR_BAR_HEIGHT,
            color=color.RED,
        )
        self.offset: int = index * (INDICATOR_BAR_HEIGHT + INDICATOR_BAR_BORDER_SIZE)
        spritelist.append(self.background_box)
        spritelist.append(self.actual_bar)

    def on_update(self: IndicatorBar, _: float) -> None:
        """Update the indicator bar."""
        # Calculate the new width of the indicator bar
        self.actual_bar.width = (
            self.target_component.get_value()
            / self.target_component.get_max_value()
            * INDICATOR_BAR_WIDTH
        )

        # Calculate the new position of the indicator bar
        self.background_box.position = self.actual_bar.position = (
            self.target_sprite.center_x,
            self.target_sprite.top + INDICATOR_BAR_DISTANCE + self.offset,
        )
        self.actual_bar.left = self.actual_bar.center_x - (INDICATOR_BAR_WIDTH / 2)
