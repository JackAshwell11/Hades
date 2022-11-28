"""Creates a shop for upgrades and special attributes/items."""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade.gui

# Custom
from hades.views.base_view import BaseView

if TYPE_CHECKING:
    from arcade.gui.events import UIOnClickEvent

    from hades.game_objects.attributes import UpgradablePlayerSection
    from hades.game_objects.players import Player

__all__ = (
    "SectionUpgradeButton",
    "ShopView",
)

# Get the logger
logger = logging.getLogger(__name__)


class SectionUpgradeButton(arcade.gui.UIFlatButton):
    """A button which will upgrade a player section if the player has enough money."""

    section_ref: UpgradablePlayerSection | None = None

    def on_click(self, _: UIOnClickEvent) -> None:
        """Upgrade a player attribute section."""
        # Make sure variables needed are valid
        assert self.section_ref is not None

        # Upgrade the section if possible
        if self.section_ref.upgrade_section():
            self.text = (
                f"{self.section_ref.attribute_section_type.name} -"
                f" {self.section_ref.next_level_cost}"
            )

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return (
            f"<SectionUpgradeButton (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )


class ShopView(BaseView):
    """
    Displays the shop UI so the player can upgrade their attributes.

    Parameters
    ----------
    player: Player
        The player object used for accessing the inventory.
    """

    def __init__(self, player: Player) -> None:
        super().__init__()
        self.player: Player = player
        vertical_box: arcade.gui.UIBoxLayout = arcade.gui.UIBoxLayout()

        # Create all the section upgrade buttons based on the amount of sections the
        # player has
        for upgradable_player_section in self.player.upgrade_sections:
            upgrade_section_button = SectionUpgradeButton(
                text=(
                    f"{upgradable_player_section.attribute_section_type.name} -"
                    f" {upgradable_player_section.next_level_cost}"
                ),
                width=200,
            )
            upgrade_section_button.section_ref = upgradable_player_section
            vertical_box.add(upgrade_section_button.with_space_around(bottom=20))

        # Create the back button
        self.add_back_button(vertical_box)

        # Register the UI elements
        self.ui_manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x", anchor_y="center_y", child=vertical_box
            )
        )

    def on_draw(self) -> None:
        """Render the screen."""
        # Clear the screen
        self.clear()

        # Draw the UI elements
        self.ui_manager.draw()

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<ShopView (Current window={self.window})>"
