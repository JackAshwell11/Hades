"""Manages creation and updates of entity progress bars."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final

# Pip
from arcade import color, get_window
from arcade.gui import UIAnchorLayout, UISpace

# Custom
from hades import UI_PADDING

if TYPE_CHECKING:
    from arcade.types.color import RGBA255

    from hades.sprite import HadesSprite
    from hades_extensions.ecs.components import Stat

__all__ = ("PROGRESS_BAR_HEIGHT", "PROGRESS_BAR_WIDTH", "ProgressBar")

# The height of the progress bar
PROGRESS_BAR_HEIGHT: Final[int] = 10 + UI_PADDING

# The width of the progress bar
PROGRESS_BAR_WIDTH: Final[int] = 50 + UI_PADDING


class ProgressBar(UIAnchorLayout):
    """Represents a variable bar that can display information about a game object.

    Attributes:
        actual_bar: The actual bar of the progress bar.
    """

    __slots__ = (
        "actual_bar",
        "order",
        "target",
        "target",
    )

    def __init__(
        self: ProgressBar,
        target: tuple[HadesSprite, type[Stat]],
        scale: tuple[float, float],
        colour: RGBA255,
        order: int,
    ) -> None:
        """Initialise the object.

        Args:
            target: The target sprite and component to display.
            scale: The scale of the progress bar.
            colour: The colour of the progress bar.
            order: The order of the progress bar in the layout.

        Raises:
            ValueError: If the scale is less than or equal to 0.
        """
        if any(s <= 0 for s in scale):
            error = "Scale must be greater than 0"
            raise ValueError(error)
        if order < 0:
            error = "Order must be greater than or equal to 0"
            raise ValueError(error)

        super().__init__(
            width=(PROGRESS_BAR_WIDTH + UI_PADDING) * scale[0],
            height=(PROGRESS_BAR_HEIGHT + UI_PADDING) * scale[1],
            size_hint=None,
        )
        self.target: tuple[HadesSprite, type[Stat]] = target
        self.actual_bar: UISpace = UISpace(color=colour)
        self.order: int = order

        # Add the actual bar to the layout making sure shrinking happens from the right
        self.with_background(color=color.BLACK)
        self.with_padding(all=UI_PADDING)
        self.add(self.actual_bar, anchor_x="left", anchor_y="top")

    def on_update(self: ProgressBar, _: float) -> None:
        """Process progress bar update logic."""
        component: Stat = get_window().model.registry.get_component(
            self.target[0].game_object_id,
            self.target[1],
        )
        value = component.get_value() / component.get_max_value()
        self.actual_bar.size_hint = (value, 1)
        self.actual_bar.visible = value > 0

    def __repr__(self: ProgressBar) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<ProgressBar (Target component={self.target_component})>"
