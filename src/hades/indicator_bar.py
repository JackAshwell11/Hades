"""Manages creation and updates of entity indicator bars."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final

# Pip
from arcade import SpriteSolidColor, color, get_window

# Custom
from hades_extensions.ecs.components import Armour, Health

if TYPE_CHECKING:
    from arcade import Sprite, SpriteList
    from arcade.types.color import RGBA255

    from hades.sprite import HadesSprite
    from hades_extensions.ecs import ComponentBase
    from hades_extensions.ecs.components import Stat

__all__ = ("INDICATOR_BAR_COMPONENTS", "IndicatorBar")

# Constants
INDICATOR_BAR_COMPONENTS: Final[set[type[ComponentBase]]] = {Armour, Health}
INDICATOR_BAR_BORDER_SIZE: Final[int] = 4
INDICATOR_BAR_DISTANCE: Final[int] = 16
INDICATOR_BAR_HEIGHT: Final[int] = 10
INDICATOR_BAR_WIDTH: Final[int] = 50
INDICATOR_BAR_COLOURS: Final[dict[type[ComponentBase], RGBA255]] = {
    Armour: color.SILVER,
    Health: color.RED,
}


class IndicatorBar:
    """Represents a variable bar that can display information about a game object.

    Attributes:
        actual_bar: The actual bar of the indicator bar.
        background_box: The background box for the indicator bar.
        fixed_position: Whether the position of the indicator bar is fixed or not.
        offset: The offset of the indicator bar relative to the other indicator bars.
    """

    __slots__ = (
        "actual_bar",
        "background_box",
        "fixed_position",
        "offset",
        "target_component",
        "target_sprite",
    )

    def __init__(
        self: IndicatorBar,
        targets: tuple[HadesSprite, Stat],
        spritelist: SpriteList[Sprite],
        index: int = 0,
        *,
        fixed_position: bool,
    ) -> None:
        """Initialise the object.

        Args:
            targets: The sprite and component to attach an indicator bar to.
            spritelist: The spritelist to add the indicator bar sprites to.
            index: The index of the indicator bar relative to the other indicator bars.
            fixed_position: Whether the position of the indicator bar is fixed or not.

        Raises:
            ValueError: If the index is less than 0.
        """
        if index < 0:
            error = f"Index must be greater than or equal to 0, got {index}."
            raise ValueError(error)
        self.target_sprite: HadesSprite = targets[0]
        self.target_component: Stat = targets[1]
        self.background_box: SpriteSolidColor = SpriteSolidColor(
            INDICATOR_BAR_WIDTH + INDICATOR_BAR_BORDER_SIZE,
            INDICATOR_BAR_HEIGHT + INDICATOR_BAR_BORDER_SIZE,
            color=color.BLACK,
        )
        self.actual_bar: SpriteSolidColor = SpriteSolidColor(
            INDICATOR_BAR_WIDTH,
            INDICATOR_BAR_HEIGHT,
            color=INDICATOR_BAR_COLOURS[type(targets[1])],
        )
        self.offset: int = index * (INDICATOR_BAR_HEIGHT + INDICATOR_BAR_BORDER_SIZE)
        self.fixed_position: bool = fixed_position
        spritelist.append(self.background_box)
        spritelist.append(self.actual_bar)

        # Set the position of the indicator bar
        if self.fixed_position:
            self.background_box.position = self.actual_bar.position = (
                INDICATOR_BAR_WIDTH / 2 + INDICATOR_BAR_BORDER_SIZE,
                get_window().height
                - INDICATOR_BAR_HEIGHT / 2
                - INDICATOR_BAR_BORDER_SIZE
                - self.offset,
            )

    def update(self: IndicatorBar) -> None:
        """Update the indicator bar."""
        # Calculate the new width of the indicator bar
        self.actual_bar.width = (
            self.target_component.get_value()
            / self.target_component.get_max_value()
            * INDICATOR_BAR_WIDTH
        )

        # Calculate the new position of the indicator bar
        if not self.fixed_position:
            self.background_box.position = self.actual_bar.position = (
                self.target_sprite.center_x,
                self.target_sprite.top + INDICATOR_BAR_DISTANCE + self.offset,
            )
        self.actual_bar.left = self.actual_bar.center_x - (INDICATOR_BAR_WIDTH / 2)

    def __repr__(self: IndicatorBar) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<IndicatorBar (Target sprite={self.target_sprite}) (Target"
            f"component={self.target_component}) (Offset={self.offset})>"
        )
