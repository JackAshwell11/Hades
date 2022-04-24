from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade.gui

# Custom
from game.constants.entity import LevelType

if TYPE_CHECKING:
    from game.views.game import Game
    from game.window import Window

# Get the logger
logger = logging.getLogger(__name__)


class Test(arcade.gui.UIFlatButton):
    def on_click(self, event):
        window: Window = arcade.get_window()
        current_view: Game = window.views["Game"]  # noqa
        current_view.player.levels[LevelType.HEALTH].test()


class BackButton(arcade.gui.UIFlatButton):
    """A button which will switch back to the game view."""

    def __repr__(self) -> str:
        return (
            f"<BackButton (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )

    def on_click(self, _) -> None:
        """Called when the button is clicked."""
        # Get the current window and view
        window: Window = arcade.get_window()
        current_view: ShopView = window.current_view  # noqa

        # Deactivate the UI manager so the buttons can't be clicked
        current_view.manager.disable()

        # Show the game view
        game_view: Game = window.views["Game"]  # noqa
        window.show_view(game_view)

        logger.info("Switching from shop view to game view")


class ShopView(arcade.View):
    """
    Displays the shop UI so the player can upgrade their attributes

    Attributes
    ----------
    manager: arcade.gui.UIManager
        Manages all the different UI elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manager: arcade.gui.UIManager = arcade.gui.UIManager()
        self.vertical_box: arcade.gui.UIBoxLayout = arcade.gui.UIBoxLayout()

        mj = Test(text="health", width=200)
        self.vertical_box.add(mj.with_space_around(top=20))

        # Create the back button
        back_button = BackButton(text="Back", width=200)
        self.vertical_box.add(back_button.with_space_around(top=20))

        # Register the UI elements
        self.manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x", anchor_y="center_y", child=self.vertical_box
            )
        )

    def __repr__(self) -> str:
        return f"<ShopView (Current window={self.window})>"

    def on_show(self) -> None:
        """Called when the view loads."""
        # Set the background color
        self.window.background_color = arcade.color.BABY_BLUE

        logger.info("Shown shop view")

    def on_draw(self) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the UI elements
        self.manager.draw()
