"""Manages creation and updates of entity progress bars."""

from __future__ import annotations

# Builtin
import operator
from typing import TYPE_CHECKING, Final, cast

# Pip
from arcade import color
from arcade.gui import UIAnchorLayout, UIBoxLayout, UISpace

# Custom
from hades_extensions.ecs.components import Stat

if TYPE_CHECKING:
    from arcade.types.color import RGBA255

    from hades.sprite import HadesSprite

__all__ = (
    "PROGRESS_BAR_DISTANCE",
    "ProgressBar",
    "ProgressBarGroup",
)

# Constants
PROGRESS_BAR_BORDER_SIZE: Final[int] = 4
PROGRESS_BAR_HEIGHT: Final[int] = 10
PROGRESS_BAR_DISTANCE: Final[int] = 36
PROGRESS_BAR_WIDTH: Final[int] = 50


class ProgressBar(UIAnchorLayout):
    """Represents a variable bar that can display information about a game object.

    Attributes:
        actual_bar: The actual bar of the progress bar.
    """

    __slots__ = (
        "actual_bar",
        "target_component",
        "target_sprite",
    )

    def __init__(
        self: ProgressBar,
        target_component: Stat,
        scale: float,
        colour: RGBA255,
    ) -> None:
        """Initialise the object.

        Args:
            target_component: The component to get the information from.
            scale: The scale of the progress bar.
            colour: The colour of the progress bar.

        Raises:
            ValueError: If the scale is less than or equal to 0.
        """
        if scale <= 0:
            error = "Scale must be greater than 0"
            raise ValueError(error)

        super().__init__(
            width=(PROGRESS_BAR_WIDTH + PROGRESS_BAR_BORDER_SIZE) * scale,
            height=(PROGRESS_BAR_HEIGHT + PROGRESS_BAR_BORDER_SIZE) * scale,
            size_hint=None,
        )
        self.target_component: Stat = target_component
        self.actual_bar: UISpace = UISpace(color=colour)

        # Add the actual bar to the layout with a small border making sure shrinking
        # happens from the right
        self.add(self.actual_bar, anchor_x="left", anchor_y="top")

    def __repr__(self: ProgressBar) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<ProgressBar (Target component={self.target_component})>"


class ProgressBarGroup(UIBoxLayout):
    """Represents a group of progress bars for a sprite's components."""

    __slots__ = ("sprite",)

    def __init__(self: ProgressBarGroup, sprite: HadesSprite) -> None:
        """Initialise the object.

        Args:
            sprite: The sprite associated with the progress bars.
        """
        super().__init__(space_between=PROGRESS_BAR_BORDER_SIZE)
        self.with_padding(all=PROGRESS_BAR_BORDER_SIZE)
        self.with_background(color=color.BLACK)
        self.sprite: HadesSprite = sprite

        # Create a progress bar for each component that supports one
        progress_bars = [
            (
                sprite.constructor.progress_bars[type(component)][0],
                ProgressBar(
                    cast(Stat, component),
                    *sprite.constructor.progress_bars[type(component)][1:],
                ),
            )
            for component in sprite.constructor.components
            if type(component) in sprite.constructor.progress_bars
        ]
        for _, progress_bar in sorted(progress_bars, key=operator.itemgetter(0)):
            self.add(progress_bar)

    def on_update(self: ProgressBarGroup, _: float) -> None:
        """Process progress bar update logic."""
        for progress_bar in self.children:  # type: ProgressBar
            value = (
                progress_bar.target_component.get_value()
                / progress_bar.target_component.get_max_value()
            )
            progress_bar.actual_bar.size_hint = (value, 1)
            progress_bar.actual_bar.visible = value > 0

    def __repr__(self: ProgressBarGroup) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<ProgressBarGroup (Target sprite={self.sprite})>"
