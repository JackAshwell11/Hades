"""Manages the player flow and registry callbacks."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final, cast

# Pip
from arcade.resources import resolve

# Custom
from hades.player.view import InventoryItemButton, ShopItemButton
from hades_extensions.ecs import EventType
from hades_extensions.ecs.components import PythonSprite
from hades_extensions.ecs.systems import ShopSystem

if TYPE_CHECKING:
    from pathlib import Path

    from arcade.gui import UIOnClickEvent

    from hades.model import GameModel
    from hades.player.view import PlayerView

__all__ = ("PlayerController",)

# The path to the shop offerings JSON file
SHOP_OFFERINGS: Final[Path] = resolve(":resources:shop_offerings.json")


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
        self.model.game_engine.setup_shop(str(SHOP_OFFERINGS))
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

    def on_shop_item_loaded(
        self: PlayerController,
        index: int,
        data: tuple[str, str, str],
        cost: int,
    ) -> None:
        """Handle the shop item loaded event.

        Args:
            index: The index of the item in the shop.
            data: A tuple containing the name, description, and icon type of the item.
            cost: The cost of the item.
        """
        self.view.player_attributes_layout.shop_layout.add_item(
            ShopItemButton(index, data, cost),
        )

    def on_shop_item_purchased(self: PlayerController, index: int, cost: int) -> None:
        """Handle the shop item purchased event.

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
        elif isinstance(item_button, ShopItemButton):
            self.model.registry.get_system(ShopSystem).purchase(
                self.model.player_id,
                item_button.shop_index,
            )
