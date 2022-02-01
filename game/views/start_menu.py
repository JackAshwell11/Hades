from __future__ import annotations

# Pip
import arcade
import arcade.gui

# Custom
from views.game import Game


class StartButton(arcade.gui.UIFlatButton):
    """A button which when clicked will start the game."""

    def on_click(self, event: arcade.gui.UIOnClickEvent) -> None:
        """Called when the button is clicked."""
        # Set up the new game
        new_game = Game(True)
        new_game.setup(1)

        # Get the current window and view
        window = arcade.get_window()
        current_view: StartMenu = window.current_view  # noqa

        # Deactivate the UI manager so the buttons can't be clicked
        current_view.manager.disable()

        # Show the game view
        window.show_view(new_game)

    def __repr__(self) -> str:
        return (
            f"<StartButton (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )


class QuitButton(arcade.gui.UIFlatButton):
    """A button which when clicked will quit the game."""

    def on_click(self, event: arcade.gui.UIOnClickEvent) -> None:
        """Called when the button is clicked."""
        arcade.exit()

    def __repr__(self) -> str:
        return (
            f"<QuitButton (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )


class StartMenu(arcade.View):
    """Creates a start menu allowing the player to pick which game mode and options they
    want."""

    def __init__(self) -> None:
        super().__init__()
        self.manager: arcade.gui.UIManager = arcade.gui.UIManager()
        self.vertical_box: arcade.gui.UIBoxLayout = arcade.gui.UIBoxLayout()

        # Create the start button
        start_button = StartButton(text="Start Game", width=200)
        self.vertical_box.add(start_button.with_space_around(bottom=20))

        # Create the quit button
        quit_button = QuitButton(text="Quit Game", width=200)
        self.vertical_box.add(quit_button.with_space_around(bottom=20))

        # Register the UI elements
        self.manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x", anchor_y="center_y", child=self.vertical_box
            )
        )

        # Enable the UI elements
        self.manager.enable()

    def __repr__(self) -> str:
        return f"<StartMenu (Current window={self.window})>"

    def on_draw(self) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the background colour
        arcade.set_background_color(arcade.color.OCEAN_BOAT_BLUE)

        # Draw the UI elements
        self.manager.draw()
