"""Manages the player flow and registry callbacks."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

# Pip
from arcade import get_default_texture

# Custom
from hades.player.view import InventoryItemButton, UpgradesItemButton
from hades_extensions.ecs import EventType
from hades_extensions.ecs.components import PythonSprite, Upgrades
from hades_extensions.ecs.systems import UpgradeSystem

if TYPE_CHECKING:
    from arcade.gui import UIOnClickEvent

    from hades.model import GameModel
    from hades.player.view import PlayerView

__all__ = ("PlayerController",)


class PlayerController:
    """Manages the player flow and registry callbacks.

    Attributes:
        model: The model managing the player state.
    """

    __slots__ = ("__weakref__", "model", "view")

    def __init__(self: PlayerController, view: PlayerView) -> None:
        """Initialise the object.

        Args:
            view: The renderer for the player.
        """
        self.model: GameModel = cast("GameModel", None)
        self.view: PlayerView = view

    def setup(self: PlayerController) -> None:
        """Set up the controller."""
        self.model.registry.add_callback(
            EventType.InventoryUpdate,
            self.on_update_inventory,
        )
        self.view.window.register_event_type("on_texture_button_callback")
        self.view.window.register_event_type("on_use_button_callback")
        self.view.window.register_event_type("on_inventory_use_item")
        self.view.window.push_handlers(self)

        upgrades = self.model.registry.get_component(
            self.model.player_id,
            Upgrades,
        ).upgrades
        for target_component, target_functions in upgrades.items():
            upgrades_item_button = UpgradesItemButton(
                target_component,
                target_functions,
            )
            upgrades_item_button.texture = get_default_texture()
            self.view.player_attributes_layout.upgrades_layout.add_item(
                upgrades_item_button,
            )

    def show_view(self: PlayerController) -> None:
        """Process show view functionality."""
        self.view.ui.enable()

    def hide_view(self: PlayerController) -> None:
        """Process hide view functionality."""
        self.view.ui.disable()

    def on_update_inventory(self: PlayerController, items: list[int]) -> None:
        """Handle the inventory update event.

        Args:
            items: The list of items in the inventory.
        """
        self.view.player_attributes_layout.inventory_layout.items = [
            InventoryItemButton(
                self.model.registry.get_component(item_id, PythonSprite).sprite,
            )
            for item_id in items
        ]

    def on_texture_button_callback(
        self: PlayerController,
        event: UIOnClickEvent,
    ) -> None:
        """Handle the texture button callback.

        Args:
            event: The event that occurred.
        """
        self.view.stats_layout.set_info(*event.source.parent.get_info())

    def on_use_button_callback(self: PlayerController, event: UIOnClickEvent) -> None:
        """Handle the use button callback.

        Args:
            event: The event that occurred.
        """
        item_button = event.source.parent
        if isinstance(item_button, InventoryItemButton):
            self.model.game_engine.use_item(
                self.model.player_id,
                item_button.sprite_object.game_object_id,
            )
        elif isinstance(item_button, UpgradesItemButton):
            self.model.registry.get_system(UpgradeSystem).upgrade_component(
                self.model.player_id,
                item_button.target_component,
            )
            self.view.stats_layout.set_info(*item_button.get_info())
