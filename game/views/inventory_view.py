from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import arcade
import arcade.gui

# Custom
from constants import CONSUMABLES

if TYPE_CHECKING:
    from entities.base import Item
    from entities.inventory import Inventory
    from views.game import Game
    from window import Window


class InventoryBox(arcade.gui.UITextureButton):
    """Represents an individual box showing an item in the player's inventory."""

    # Class variables
    item_ref: Item | None = None

    def __repr__(self) -> str:
        return (
            f"<InventoryBox (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )

    def on_click(self, _) -> None:
        """Called when the button is clicked."""
        # Stop slots being clicked on if they're empty
        if not self.item_ref:
            return

        # Check if the item is a consumable
        if self.item_ref.item_id in CONSUMABLES:
            # Use it
            self.item_ref.item_use()

            # Get the current window, current view and game view
            window: Window = arcade.get_window()
            game_view: Game = window.views["Game"]  # noqa

            # Remove the item from the player's inventory and clear the texture and item
            # ref
            game_view.player.inventory.remove(self.item_ref)
            self.item_ref = self.texture = None

            # Update the button
            self.trigger_full_render()


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
        for i in range(self.inventory.height):
            horizontal_box = arcade.gui.UIBoxLayout(vertical=False)
            for j in range(self.inventory.width):
                horizontal_box.add(
                    InventoryBox(width=128, height=128).with_border(width=4)
                )
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

    def on_show(self) -> None:
        """Called when the view loads."""
        # Set the background color
        self.window.background_color = arcade.color.BABY_BLUE

        # Update each box to show the player's inventory
        for row_count, box_layout in enumerate(self.vertical_box.children):
            for column_count, ui_border_obj in enumerate(box_layout.children):
                # Check if the current object is the back button
                if not ui_border_obj.children:
                    continue

                # Get the inventory array position
                array_pos = column_count + row_count * 3

                # Check if there actually exists an item at the given position
                if len(self.inventory.array) - 1 < array_pos:
                    continue

                # Set the inventory box item reference and its texture
                inventory_box_obj: InventoryBox = ui_border_obj.children[0]  # noqa
                inventory_box_obj.item_ref = self.inventory.array[array_pos]
                inventory_box_obj.texture = self.inventory.array[array_pos].texture

    def on_draw(self) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the UI elements
        self.manager.draw()
