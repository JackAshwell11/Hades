"""Creates a start menu so the player can change their settings or game mode."""

from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade
from arcade.gui import (
    UIAnchorLayout,
    UIBoxLayout,
    UIFlatButton,
    UIManager,
    UIOnClickEvent,
)

# Custom
from hades.views.game import Game

if TYPE_CHECKING:
    from hades.window import Window

__all__ = ("StartMenu",)

# Get the logger
logger = logging.getLogger(__name__)


def start_on_click_handler(_: UIOnClickEvent) -> None:
    """Create a game instance when the button is clicked."""
    # Get the current window and view
    window: Window = arcade.get_window()

    # Set up the new game
    new_game = Game(0)
    window.views["Game"] = new_game
    logger.info("Initialised game view at level %d", 0)

    # Show the new game
    window.show_view(new_game)


def quit_on_click_handler(_: UIOnClickEvent) -> None:
    """Exit the game when the button is clicked."""
    logger.info("Exiting game")
    arcade.exit()


class StartMenu(arcade.View):
    """Creates a start menu useful for picking the game mode and options.

    Attributes:
        ui_manager: Manages all the different UI elements for this view.
    """

    def __init__(self: StartMenu) -> None:
        """Initialise the object."""
        super().__init__()
        self.ui_manager: UIManager = UIManager()

        # Create the buttons
        vertical_box = UIBoxLayout(space_between=20)
        start_button, quit_button = (
            UIFlatButton(text="Start Game", width=200),
            UIFlatButton(text="Quit Game", width=200),
        )
        start_button.on_click, quit_button.on_click = (
            start_on_click_handler,
            quit_on_click_handler,
        )
        vertical_box.add(start_button)
        vertical_box.add(quit_button)

        # Add the vertical box layout to the UI
        anchor_layout = UIAnchorLayout(anchor_x="center_x", anchor_y="center_y")
        anchor_layout.add(vertical_box)
        self.ui_manager.add(anchor_layout)

    def on_draw(self: StartMenu) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the background colour
        self.window.background_color = arcade.color.OCEAN_BOAT_BLUE

        # Draw the UI elements
        self.ui_manager.draw()

    def on_show_view(self: StartMenu) -> None:
        """Process show view functionality."""
        self.ui_manager.enable()

    def on_hide_view(self: StartMenu) -> None:
        """Process hide view functionality."""
        self.ui_manager.disable()

    def __repr__(self: StartMenu) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<StartMenu (Current window={self.window})>"
