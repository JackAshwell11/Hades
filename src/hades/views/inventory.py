"""Displays the player's inventory graphically."""
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
    UITextureButton,
)

# Custom
from hades.constants_OLD.general import INVENTORY_HEIGHT, INVENTORY_WIDTH
from hades.game_objects.base import CollectibleTile

if TYPE_CHECKING:
    from arcade.gui.events import UIOnClickEvent

    from hades.game_objects.players import Player
    from hades.window import Window

__all__ = ("Inventory",)

# Get the logger
logger = logging.getLogger(__name__)


class InventoryBox(UITextureButton):
    """Represents an individual box showing an item in the player's inventory."""

    # Class variables
    item_ref: CollectibleTile | None = None

    def on_click(self: InventoryBox, _: UIOnClickEvent) -> None:
        """Use an item in the player's inventory."""
        # Stop slots being clicked on if they're empty
        if not self.item_ref:
            return

        # Check if the item can be used or not
        if issubclass(type(self.item_ref), CollectibleTile):
            # Use it
            if not self.item_ref.item_use():
                # Use was not successful
                logger.info("Item use for %r not successful", self.item_ref)
                return

            # Get the current window, current view and game view
            window: Window = arcade.get_window()
            current_view: Inventory = window.current_view  # type: ignore[assignment]

            # Remove the item from the player's inventory and clear the item ref and the
            # inventory box texture
            current_view.player.inventory.remove(self.item_ref)
            self.item_ref = self.texture = None

            # Update the grid
            current_view.update_grid()

            # Render the changes. This is needed because the view is only updated when
            # the user exits
            self.trigger_full_render()

            logger.info("Item use for %r successful", self.item_ref)

    def __repr__(self: InventoryBox) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<InventoryBox (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )


def back_on_click(_: UIOnClickEvent) -> None:
    """Return to the game when the button is clicked."""
    window = arcade.get_window()
    window.show_view(window.views["Game"])


class Inventory(arcade.View):
    """Displays the player's inventory allowing them to manage it and equip items.

    Attributes:
        ui_manager: Manages all the different UI elements for this view.
        vertical_box: The arcade box layout responsible for organising the different UI
            elements.
    """

    def __init__(self: Inventory, player: Player) -> None:
        """Initialise the object.

        Args:
            player: The player object used for accessing the inventory.
        """
        super().__init__()
        self.player: Player = player
        self.ui_manager: UIManager = UIManager()
        self.vertical_box: UIBoxLayout = UIBoxLayout(space_between=20)

        # Create the inventory grid
        for _ in range(INVENTORY_HEIGHT):
            horizontal_box = UIBoxLayout(vertical=False)
            for _ in range(INVENTORY_WIDTH):
                horizontal_box.add(
                    InventoryBox(width=64, height=64).with_border(width=4),
                )
            self.vertical_box.add(horizontal_box)

        # Create the back button
        back_button = UIFlatButton(text="Back", width=200)
        back_button.on_click = back_on_click
        self.vertical_box.add(back_button)

        # Add the vertical box layout to the UI
        anchor_layout = UIAnchorLayout(anchor_x="center_x", anchor_y="center_y")
        anchor_layout.add(self.vertical_box)
        self.ui_manager.add(anchor_layout)

    def on_draw(self: Inventory) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the UI elements
        self.ui_manager.draw()

    def update_grid(self: Inventory) -> None:
        """Update the inventory grid with the player's current inventory."""
        result = [0, 0]
        for row_count, box_layout in enumerate(self.vertical_box.children):
            for column_count, ui_border_obj in enumerate(box_layout.children):
                # Check if the current object is the back button
                if not ui_border_obj.children:
                    continue

                # Get the inventory array position
                array_pos = column_count + row_count * 3

                # Set the inventory box object to the inventory item
                inventory_box_obj: InventoryBox = ui_border_obj.children[0]
                try:
                    item = self.player.inventory[array_pos]
                    inventory_box_obj.item_ref = item
                    inventory_box_obj.texture = item.texture
                    result[0] += 1
                except IndexError:
                    inventory_box_obj.texture = None
                    result[1] += 1
        logger.info(
            "Updated inventory grid with %d item textures and %d empty textures",
            *result,
        )

    def __repr__(self: Inventory) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Inventory (Current window={self.window})>"
