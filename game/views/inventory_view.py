from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import arcade
import arcade.gui

if TYPE_CHECKING:
    from entities.inventory import Inventory
    from window import Window
    from views.game import Game


class InventoryBox(arcade.gui.UITextureButton):
    """Represents an individual box showing an item in the player's inventory."""

    def __init__(self) -> None:
        super().__init__(width=100, height=100)

    def __repr__(self) -> str:
        return (
            f"<InventoryBox (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )


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
        current_view: InventoryView = window.current_view  # noqa

        # Deactivate the UI manager so the buttons can't be clicked
        current_view.manager.disable()

        # Show the game view
        game_view: Game = window.views["Game"]  # noqa
        window.show_view(game_view)


class InventoryView(arcade.View):
    """
    Displays the player's inventory allowing them to manage it and equip items.

    Parameters
    ----------
    inventory: Inventory
        The inventory object for the player.

    Attributes
    ----------
    manager: arcade.gui.UIManager
        Manages all the different UI elements.
    """

    def __init__(self, inventory: Inventory) -> None:
        super().__init__()
        self.manager: arcade.gui.UIManager = arcade.gui.UIManager()
        self.vertical_box: arcade.gui.UIBoxLayout = arcade.gui.UIBoxLayout()
        self.inventory: Inventory = inventory

        # Create the inventory grid
        for _ in range(self.inventory.grid_size):
            horizontal_box = arcade.gui.UIBoxLayout(vertical=False)
            for _ in range(self.inventory.grid_size):
                horizontal_box.add(InventoryBox().with_border(color=arcade.color.WHITE))
            self.vertical_box.add(horizontal_box)

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
        return f"<InventoryView (Current window={self.window})>"

    def get_item(self, index: int) -> str:
        """
        Gets an item at a specific index from the inventory.

        Parameters
        ----------
        index: int
            The index to get the item at.

        Returns
        -------
        str
            The name of the item. If the item doesn't exist, this wil be None.
        """
        # Check if there actually exists an item at the given position
        if len(self.inventory.array) - 1 < index:
            return ""

        # Get the item name
        return str(self.inventory.array[index].item_id.value)

    def on_show(self) -> None:
        """Called when the view loads."""
        # Set the background color
        self.window.background_color = arcade.color.BLACK

        # Update each box to show the player's inventory
        for row_count, box_layout in enumerate(self.vertical_box.children):
            for column_count, obj in enumerate(box_layout.children):
                # Check if the current object is the back button
                if not obj.children:
                    continue

                # Set the text attribute
                obj.children[0].text = self.get_item(row_count + column_count)  # noqa

    def on_draw(self) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the UI elements
        self.manager.draw()
