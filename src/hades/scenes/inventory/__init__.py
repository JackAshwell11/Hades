"""Contains the functionality that manages the inventory menu and its events."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.scenes.base import BaseScene
from hades.scenes.inventory.view import InventoryItemButton, InventoryView
from hades_extensions.ecs import EventType
from hades_extensions.ecs.components import PythonSprite

if TYPE_CHECKING:
    from typing import ClassVar

    from arcade.gui import UIOnClickEvent

__all__ = ("InventoryScene",)


class InventoryScene(BaseScene[InventoryView]):
    """Manages the inventory menu and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[InventoryView]] = InventoryView

    def add_callbacks(self: InventoryScene) -> None:
        """Add callbacks for the scene."""
        callbacks = [
            (EventType.InventoryUpdate, self.on_update_inventory),
            (EventType.InventoryOpen, self.on_inventory_open),
        ]
        for event_type, callback in callbacks:
            self.model.registry.add_callback(  # type: ignore[call-overload]
                event_type,
                callback,
            )

    def on_hide_view(self: InventoryScene) -> None:
        """Process hide view functionality."""
        super().on_hide_view()
        self.view.grid_layout.stats_layout.reset()

    def on_update_inventory(self: InventoryScene, items: list[int]) -> None:
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

    def on_texture_button_callback(self: InventoryScene, event: UIOnClickEvent) -> None:
        """Process texture button callback logic.

        Args:
            event: The event that occurred.
        """
        self.view.grid_layout.stats_layout.set_info(*event.source.parent.get_info())

    def on_use_button_callback(self: InventoryScene, event: UIOnClickEvent) -> None:
        """Process use button callback logic.

        Args:
            event: The event that occurred.
        """
        self.model.game_engine.use_item(
            self.model.player_id,
            event.source.parent.sprite_object.game_object_id,
        )

    def on_inventory_open(self: InventoryScene) -> None:
        """Process inventory open logic."""
        self.view.window.show_view(self)
