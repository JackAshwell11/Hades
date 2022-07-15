"""Creates a start menu so the player can change their settings or game mode."""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade.gui

# Custom
from game.constants.general import DEBUG_GAME
from game.views.base_view import BaseView
from game.views.game_view import Game

if TYPE_CHECKING:
    from game.window import Window

__all__ = ("StartMenu",)

# Get the logger
logger = logging.getLogger(__name__)


class StartButton(arcade.gui.UIFlatButton):
    """A button which when clicked will start the game."""

    def on_click(self, _: arcade.gui.UIOnClickEvent) -> None:
        """Create a game instance when the button is clicked."""
        # Get the current window and view
        window: Window = arcade.get_window()

        # Set up the new game
        new_game = Game()
        window.views["Game"] = new_game
        new_game.setup(1)
        logger.info(
            "Initialised game view at level %d with debug mode %s",
            1,
            "ON" if DEBUG_GAME else "OFF",
        )

        # Show the new game
        window.show_view(new_game)

    def __repr__(self) -> str:
        return (
            f"<StartButton (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )


class QuitButton(arcade.gui.UIFlatButton):
    """A button which when clicked will quit the game."""

    def on_click(self, _: arcade.gui.UIOnClickEvent) -> None:
        """Exit the game when the button is clicked."""
        logger.info("Exiting game")
        arcade.exit()

    def __repr__(self) -> str:
        return (
            f"<QuitButton (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )


class StartMenu(BaseView):
    """Creates a start menu useful for picking the game mode and options."""

    def __init__(self) -> None:
        super().__init__()
        vertical_box: arcade.gui.UIBoxLayout = arcade.gui.UIBoxLayout()

        # Create the start button
        start_button = StartButton(text="Start Game", width=200)
        vertical_box.add(start_button.with_space_around(bottom=20))

        # Create the quit button
        quit_button = QuitButton(text="Quit Game", width=200)
        vertical_box.add(quit_button.with_space_around(bottom=20))

        # Register the UI elements
        self.ui_manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x", anchor_y="center_y", child=vertical_box
            )
        )

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<StartMenu (Current window={self.window})>"

    def on_draw(self) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the background colour
        self.window.background_color = arcade.color.OCEAN_BOAT_BLUE

        # Draw the UI elements
        self.ui_manager.draw()
