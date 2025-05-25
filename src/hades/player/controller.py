"""Manages the player flow and registry callbacks."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

# Pip
from arcade import get_default_texture

# Custom
from hades_extensions.ecs import EventType
from hades_extensions.ecs.components import (
    Inventory,
    InventorySize,
    PythonSprite,
    Upgrades,
)

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
        self.view.player_attributes_layout.inventory_layout.total_size = int(
            self.model.registry.get_component(
                self.model.player_id,
                InventorySize,
            ).get_value(),
        )
        self.view.player_attributes_layout.upgrades_layout.total_size = len(
            self.model.registry.get_component(self.model.player_id, Upgrades).upgrades,
        )
        self.view.window.register_event_type("on_texture_button_callback")
        self.view.window.register_event_type("on_use_button_callback")
        self.view.window.register_event_type("on_inventory_use_item")
        self.view.window.push_handlers(self)

        upgrades = self.model.registry.get_component(
            self.model.player_id,
            Upgrades,
        ).upgrades
        for index, (target_component, target_functions) in enumerate(upgrades.items()):
            upgrades_item_button = (
                self.view.player_attributes_layout.upgrades_layout.items[index]
            )
            upgrades_item_button.target_component = target_component
            upgrades_item_button.target_functions = target_functions
            upgrades_item_button.texture = get_default_texture()

    def show_view(self: PlayerController) -> None:
        """Process show view functionality."""
        self.view.ui.enable()

    def hide_view(self: PlayerController) -> None:
        """Process hide view functionality."""
        self.view.ui.disable()

    def on_update_inventory(self: PlayerController, _: int) -> None:
        """Update the inventory view."""
        inventory_items = self.model.registry.get_component(
            self.model.player_id,
            Inventory,
        ).items
        for index, button in enumerate(
            self.view.player_attributes_layout.inventory_layout.items,
        ):
            if index < len(inventory_items):
                button.sprite_object = self.model.registry.get_component(
                    inventory_items[index],
                    PythonSprite,
                ).sprite
            else:
                button.sprite_object = None

    def on_texture_button_callback(
        self: PlayerController,
        event: UIOnClickEvent,
    ) -> None:
        """Handle the texture button callback.

        Args:
            event: The event that occurred.
        """
        self.view.stats_layout.set_info(*event.source.parent.parent.get_info())

    def on_use_button_callback(self: PlayerController, event: UIOnClickEvent) -> None:
        """Handle the use button callback.

        Args:
            event: The event that occurred.
        """
        _ = self  # Ignore linting errors
        event.source.parent.parent.use()
