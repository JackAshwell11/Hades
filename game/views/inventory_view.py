from __future__ import annotations

# Pip
import arcade


class InventoryView(arcade.View):
    """Displays the player's inventory allowing them to manage it and equip items."""

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return f"<InventoryView (Current window={self.window})>"

    def on_show(self) -> None:
        """Called when the view loads."""
        # Set the background color
        self.window.background_color = arcade.color.BLACK

    def on_draw(self) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()
