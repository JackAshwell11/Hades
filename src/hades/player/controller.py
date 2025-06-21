"""Manages the player flow and registry callbacks."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.player.view import InventoryItemButton
from hades_extensions.ecs import EventType
from hades_extensions.ecs.components import PythonSprite

if TYPE_CHECKING:
    from arcade.gui import UIOnClickEvent

    from hades.model import HadesModel
    from hades.player.view import PlayerView

__all__ = ("PlayerController",)


class PlayerController:
    """Manages the player flow and registry callbacks."""

    __slots__ = ("__weakref__", "model", "view")

    def __init__(self: PlayerController, model: HadesModel, view: PlayerView) -> None:
        """Initialise the object.

        Args:
            model: The model providing access to the game engine and its functionality.
            view: The renderer for the player.
        """
        self.model: HadesModel = model
        self.view: PlayerView = view

    def add_callbacks(self: PlayerController) -> None:
        """Set up the controller callbacks."""
        self.model.registry.add_callback(
            EventType.InventoryUpdate,
            self.on_update_inventory,
        )
        self.view.window.register_event_type("on_texture_button_callback")
        self.view.window.register_event_type("on_use_button_callback")

    def show_view(self: PlayerController) -> None:
        """Process show view functionality."""
        self.view.ui.enable()
        self.view.window.push_handlers(self)

    def hide_view(self: PlayerController) -> None:
        """Process hide view functionality."""
        self.view.ui.disable()
        self.view.grid_layout.stats_layout.reset()
        self.view.window.remove_handlers(self)

    def on_update_inventory(self: PlayerController, items: list[int]) -> None:
        """Process inventory update logic.

        Args:
            items: The list of items in the inventory.
        """
        self.view.grid_layout.items = [
            InventoryItemButton(
                self.model.registry.get_component(item_id, PythonSprite).sprite,
            )
            for item_id in items
        ]

    def on_texture_button_callback(
        self: PlayerController,
        event: UIOnClickEvent,
    ) -> None:
        """Process texture button callback logic.

        Args:
            event: The event that occurred.
        """
        self.view.grid_layout.stats_layout.set_info(*event.source.parent.get_info())

    def on_use_button_callback(self: PlayerController, event: UIOnClickEvent) -> None:
        """Process use button callback logic.

        Args:
            event: The event that occurred.
        """
        self.model.game_engine.use_item(
            self.model.player_id,
            event.source.parent.sprite_object.game_object_id,
        )
