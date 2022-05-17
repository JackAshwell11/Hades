from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade.gui

# Custom
from game.views.base import BaseView

if TYPE_CHECKING:
    from game.entities.player import Player, UpgradableSection
    from game.views.game import Game
    from game.window import Window

# Get the logger
logger = logging.getLogger(__name__)


class SectionUpgradeButton(arcade.gui.UIFlatButton):
    """A button which will upgrade a player section if the player has enough money."""

    section_ref: UpgradableSection | None = None

    def on_click(self, _) -> None:
        """Called when the button is clicked."""
        # Make sure variables needed are valid
        assert self.section_ref is not None

        # Upgrade the section if it is possible
        self.section_ref.upgrade(self)


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

        # Show the game view
        game_view: Game = window.views["Game"]  # noqa
        window.show_view(game_view)

        logger.info("Switching from shop view to game view")


class ShopView(BaseView):
    """
    Displays the shop UI so the player can upgrade their attributes

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
        for (
            upgrade_type,
            upgradable_section_obj,
        ) in self.player.upgrade_sections.items():
            upgrade_section_button = SectionUpgradeButton(
                text=f"{upgrade_type.value} - {upgradable_section_obj.next_level_cost}",
                width=200,
            )
            upgrade_section_button.section_ref = upgradable_section_obj
            vertical_box.add(upgrade_section_button.with_space_around(bottom=20))

        # Create the back button
        back_button = BackButton(text="Back", width=200)
        vertical_box.add(back_button)

        # Register the UI elements
        self.ui_manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x", anchor_y="center_y", child=vertical_box
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
        self.ui_manager.draw()
