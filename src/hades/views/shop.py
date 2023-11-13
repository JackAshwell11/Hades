"""Creates a shop for upgrades and special attributes/items."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import arcade
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIFlatButton, UIManager

if TYPE_CHECKING:
    from arcade.gui.events import UIOnClickEvent

    from hades.game_objects.attributes import UpgradablePlayerSection
    from hades.game_objects.players import Player

__all__ = (
    "SectionUpgradeButton",
    "Shop",
)


class SectionUpgradeButton(UIFlatButton):
    """A button which will upgrade a player section if the player has enough money."""

    section_ref: UpgradablePlayerSection | None = None

    def on_click(self: SectionUpgradeButton, _: UIOnClickEvent) -> None:
        """Upgrade a player attribute section."""
        # Make sure variables needed are valid
        assert self.section_ref is not None

        # Upgrade the section if possible
        if self.section_ref.upgrade_section():
            self.text = (
                f"{self.section_ref.section_type.name} -"
                f" {self.section_ref.next_level_cost}"
            )

    def __repr__(self: SectionUpgradeButton) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<SectionUpgradeButton (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )


def back_on_click(_: UIOnClickEvent) -> None:
    """Return to the game when the button is clicked."""
    window = arcade.get_window()
    window.show_view(window.views["Game"])


class Shop(arcade.View):
    """Display the shop UI so the player can upgrade their attributes.

    Attributes:
        ui_manager: Manages all the different UI elements for this view.
    """

    def __init__(self: Shop, player: Player) -> None:
        """Initialise the object.

        Args:
            player: The player object used for accessing the inventory.
        """
        super().__init__()
        self.player: Player = player
        self.ui_manager: UIManager = UIManager()

        # Create all the section upgrade buttons based on the amount of sections the
        # player has
        vertical_box: UIBoxLayout = UIBoxLayout(space_between=20)
        for upgradable_player_section in self.player.upgrade_sections:
            upgrade_section_button = SectionUpgradeButton(
                text=(
                    f"{upgradable_player_section.section_type.name} -"
                    f" {upgradable_player_section.next_level_cost}"
                ),
                width=200,
            )
            upgrade_section_button.section_ref = upgradable_player_section
            vertical_box.add(upgrade_section_button)

        # Create the back button
        back_button = UIFlatButton(text="Back", width=200)
        back_button.on_click = back_on_click
        vertical_box.add(back_button)

        # Add the vertical box layout to the UI
        anchor_layout = UIAnchorLayout(anchor_x="center_x", anchor_y="center_y")
        anchor_layout.add(vertical_box)
        self.ui_manager.add(anchor_layout)

    def on_draw(self: Shop) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the UI elements
        self.ui_manager.draw()

    def on_show_view(self: Shop) -> None:
        """Process show view functionality."""
        self.ui_manager.enable()

    def on_hide_view(self: Shop) -> None:
        """Process hide view functionality."""
        self.ui_manager.disable()

    def __repr__(self: Shop) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Shop (Current window={self.window})>"
