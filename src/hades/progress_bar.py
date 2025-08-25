"""Manages creation and updates of entity progress bars."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final

# Pip
from arcade import color
from arcade.gui import UIAnchorLayout, UISpace

# Custom
from hades import UI_PADDING

if TYPE_CHECKING:
    from arcade.types.color import RGBA255

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

    __slots__ = ("actual_bar", "order")

    def __init__(
        self: ProgressBar,
        scale: tuple[float, float],
        colour: RGBA255,
        order: int,
    ) -> None:
        """Initialise the object.

        Args:
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
        self.actual_bar: UISpace = UISpace(color=colour)
        self.order: int = order
        self.actual_bar.size_hint = (1, 1)

        # Add the actual bar to the layout making sure shrinking happens from the right
        self.with_background(color=color.BLACK)
        self.with_padding(all=UI_PADDING)
        self.add(self.actual_bar, anchor_x="left", anchor_y="top")
