from __future__ import annotations

# Pip
import arcade

# Custom
from views.start_menu import StartMenu


class Window(arcade.Window):
    """
    Manages the window and allows switching between views.

     Attributes
    ----------
    views: dict[str, arcade.View]
        Holds all the views used by the game.
    """

    def __init__(self) -> None:
        super().__init__()
        self.views: dict[str, arcade.View] = {}

    def __repr__(self) -> str:
        return f"<Window (Width={self.width}) (Height={self.height})>"


def main() -> None:
    """Initialises the game and runs it."""
    # Initialise the window
    window = Window()
    window.center_window()

    # Initialise and load the start menu view
    new_view = StartMenu()
    window.views["StartMenu"] = new_view
    window.show_view(window.views["StartMenu"])
    new_view.manager.enable()

    # Run the game
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
