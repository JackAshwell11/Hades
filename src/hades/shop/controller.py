"""Manages the shop flow and registry callbacks."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades import ViewType
from hades.shop.view import ShopItemButton
from hades_extensions.ecs import EventType
from hades_extensions.ecs.systems import ShopSystem

if TYPE_CHECKING:
    from arcade.gui import UIOnClickEvent

    from hades.model import HadesModel
    from hades.shop.view import ShopView

__all__ = ("ShopController",)


class ShopController:
    """Manages the shop flow and registry callbacks."""

    __slots__ = ("__weakref__", "model", "view")

    def __init__(self: ShopController, model: HadesModel, view: ShopView) -> None:
        """Initialise the object.

        Args:
            model: The model providing access to the game engine and its functionality.
            view: The renderer for the shop.
        """
        self.model: HadesModel = model
        self.view: ShopView = view

    def add_callbacks(self: ShopController) -> None:
        """Set up the controller callbacks."""
        callbacks = [
            (EventType.ShopItemLoaded, self.on_shop_item_loaded),
            (EventType.ShopItemPurchased, self.on_shop_item_purchased),
            (EventType.ShopOpen, self.on_shop_open),
        ]
        for event_type, callback in callbacks:
            self.model.registry.add_callback(  # type: ignore[call-overload]
                event_type,
                callback,
            )
        self.view.window.register_event_type("on_texture_button_callback")
        self.view.window.register_event_type("on_use_button_callback")

    def show_view(self: ShopController) -> None:
        """Process show view functionality."""
        self.view.ui.enable()
        self.view.window.push_handlers(self)

    def hide_view(self: ShopController) -> None:
        """Process hide view functionality."""
        self.view.ui.disable()
        self.view.grid_layout.stats_layout.reset()
        self.view.window.remove_handlers(self)

    def on_shop_item_loaded(
        self: ShopController,
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
        self.view.grid_layout.add_item(ShopItemButton(index, data, cost))

    def on_shop_item_purchased(self: ShopController, index: int, cost: int) -> None:
        """Process shop item purchased logic.

        Args:
            index: The index of the item that was purchased.
            cost: The cost of the item that was purchased.
        """
        shop_item_button = self.view.grid_layout.items[index]
        shop_item_button.cost = cost
        self.view.grid_layout.stats_layout.set_info(*shop_item_button.get_info())

    def on_texture_button_callback(
        self: ShopController,
        event: UIOnClickEvent,
    ) -> None:
        """Process texture button callback logic.

        Args:
            event: The event that occurred.
        """
        self.view.grid_layout.stats_layout.set_info(*event.source.parent.get_info())

    def on_use_button_callback(self: ShopController, event: UIOnClickEvent) -> None:
        """Process use button callback logic.

        Args:
            event: The event that occurred.
        """
        self.model.registry.get_system(ShopSystem).purchase(
            self.model.player_id,
            event.source.parent.shop_index,
        )

    def on_shop_open(self: ShopController) -> None:
        """Process shop open logic."""
        self.view.window.show_view(self.view.window.views[ViewType.SHOP])
