from __future__ import annotations

# Pip
import arcade

# Custom
from views.start_menu import StartMenu


class Window(arcade.Window):
    """Manages the window and allows switching between views."""

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return f"<Window (Width={self.width}) (Height={self.height})>"


def main() -> None:
    """Initialises the game and runs it."""
    window = Window()
    window.center_window()
    window.show_view(StartMenu())
    window.run()


if __name__ == "__main__":
    main()
