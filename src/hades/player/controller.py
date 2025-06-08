"""Manages the player flow and registry callbacks."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.player.view import InventoryItemButton, ShopItemButton
from hades_extensions.ecs import EventType
from hades_extensions.ecs.components import PythonSprite
from hades_extensions.ecs.systems import ShopSystem

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
        callbacks = [
            (EventType.InventoryUpdate, self.on_update_inventory),
            (EventType.ShopItemLoaded, self.on_shop_item_loaded),
            (EventType.ShopItemPurchased, self.on_shop_item_purchased),
        ]
        for event_type, callback in callbacks:
            self.model.registry.add_callback(  # type: ignore[call-overload]
                event_type,
                callback,
            )
        self.view.window.register_event_type("on_texture_button_callback")
        self.view.window.register_event_type("on_use_button_callback")
        self.view.window.register_event_type("on_inventory_use_item")
        self.view.window.push_handlers(self)

    def show_view(self: PlayerController) -> None:
        """Process show view functionality."""
        self.view.ui.enable()

    def hide_view(self: PlayerController) -> None:
        """Process hide view functionality."""
        self.view.ui.disable()

    def on_update_inventory(self: PlayerController, items: list[int]) -> None:
        """Process inventory update logic.

        Args:
            items: The list of items in the inventory.
        """
        self.view.player_attributes_layout.inventory_layout.items = [
            InventoryItemButton(
                self.model.registry.get_component(item_id, PythonSprite).sprite,
            )
            for item_id in items
        ]

    def on_shop_item_loaded(
        self: PlayerController,
        index: int,
        data: tuple[str, str, str],
        cost: int,
    ) -> None:
        """Process shop item loaded logic.

        Args:
            index: The index of the item in the shop.
            data: A tuple containing the name, description, and icon type of the item.
            cost: The cost of the item.
        """
        self.view.player_attributes_layout.shop_layout.add_item(
            ShopItemButton(index, data, cost),
        )

    def on_shop_item_purchased(self: PlayerController, index: int, cost: int) -> None:
        """Process shop item purchased logic.

        Args:
            index: The index of the item that was purchased.
            cost: The cost of the item that was purchased.
        """
        shop_item_button = self.view.player_attributes_layout.shop_layout.items[index]
        shop_item_button.cost = cost
        self.view.stats_layout.set_info(*shop_item_button.get_info())

    def on_texture_button_callback(
        self: PlayerController,
        event: UIOnClickEvent,
    ) -> None:
        """Process texture button callback logic.

        Args:
            event: The event that occurred.
        """
        self.view.stats_layout.set_info(*event.source.parent.get_info())

    def on_use_button_callback(self: PlayerController, event: UIOnClickEvent) -> None:
        """Process use button callback logic.

        Args:
            event: The event that occurred.
        """
        item_button = event.source.parent
        if isinstance(item_button, InventoryItemButton):
            self.model.game_engine.use_item(
                self.model.player_id,
                item_button.sprite_object.game_object_id,
            )
        elif isinstance(item_button, ShopItemButton):
            self.model.registry.get_system(ShopSystem).purchase(
                self.model.player_id,
                item_button.shop_index,
            )
