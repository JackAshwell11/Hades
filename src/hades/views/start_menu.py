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
    from hades.window import HadesWindow

__all__ = ("StartMenu",)

# Get the logger
logger = logging.getLogger(__name__)


class StartButton(UIFlatButton):
    """Represents a button that starts the game when clicked."""

    def __init__(self: StartButton) -> None:
        """Initialise the object."""
        super().__init__(text="Start Game", width=200)

    # pylint: disable=no-self-use
    def on_click(self: StartButton, _: UIOnClickEvent) -> None:
        """Create a game instance when the button is clicked."""
        # Get the current window and view
        window: HadesWindow = arcade.get_window()

        # Set up the new game
        new_game = Game()
        new_game.setup(0)
        logger.info("Initialised game view at level %d", 0)

        # Show the new game
        window.show_view(new_game)
        logger.debug("Showed game view")

    def __repr__(self: StartButton) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<StartButton (Text={self.text})>"


class QuitButton(UIFlatButton):
    """Represents a button that quits the game when clicked."""

    def __init__(self: QuitButton) -> None:
        """Initialise the object."""
        super().__init__(text="Quit Game", width=200)

    # pylint: disable=no-self-use
    def on_click(self: QuitButton, _: UIOnClickEvent) -> None:
        """Quit the game when the button is clicked."""
        arcade.exit()
        logger.info("Exiting game")

    def __repr__(self: QuitButton) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<QuitButton (Text={self.text})>"


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
        vertical_box.add(StartButton())
        vertical_box.add(QuitButton())

        # Add the vertical box layout to the UI
        anchor_layout = UIAnchorLayout(anchor_x="center_x", anchor_y="center_y")
        anchor_layout.add(vertical_box)
        self.ui_manager.add(anchor_layout)

    def on_draw(self: StartMenu) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the background colour and the UI elements
        self.window.background_color = arcade.color.OCEAN_BOAT_BLUE
        self.ui_manager.draw()

    def on_show_view(self: StartMenu) -> None:
        """Process show view functionality."""
        self.ui_manager.enable()
        logger.debug("Showing start menu view")

    def on_hide_view(self: StartMenu) -> None:
        """Process hide view functionality."""
        self.ui_manager.disable()
        logger.debug("Hiding start menu view")

    def __repr__(self: StartMenu) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<StartMenu (Current window={self.window})>"
