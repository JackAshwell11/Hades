"""Contains the functionality that manages the shop menu and its events."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.scenes.base import BaseScene
from hades.scenes.shop.view import ShopItemButton, ShopView
from hades_engine import EventType, add_callback
from hades_engine.ecs.systems import ShopSystem

if TYPE_CHECKING:
    from typing import ClassVar

    from arcade.gui import UIOnClickEvent

__all__ = ("ShopScene",)


class ShopScene(BaseScene[ShopView]):
    """Manages the shop menu and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[ShopView]] = ShopView

    def add_callbacks(self: ShopScene) -> None:
        """Add callbacks for the scene."""
        callbacks = [
            (EventType.ShopItemLoaded, self.on_shop_item_loaded),
            (EventType.ShopItemPurchased, self.on_shop_item_purchased),
            (EventType.ShopOpen, self.on_shop_open),
        ]
        for event_type, callback in callbacks:
            add_callback(event_type, callback)  # type: ignore[call-overload]

    def on_hide_view(self: ShopScene) -> None:
        """Process hide view functionality."""
        super().on_hide_view()
        self.view.grid_layout.stats_layout.reset()

    def on_shop_item_loaded(
        self: ShopScene,
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

    def on_shop_item_purchased(self: ShopScene, index: int, cost: int) -> None:
        """Process shop item purchased logic.

        Args:
            index: The index of the item that was purchased.
            cost: The cost of the item that was purchased.
        """
        shop_item_button = self.view.grid_layout.items[index]
        shop_item_button.cost = cost
        self.view.grid_layout.stats_layout.set_info(*shop_item_button.get_info())

    def on_texture_button_callback(self: ShopScene, event: UIOnClickEvent) -> None:
        """Process texture button callback logic.

        Args:
            event: The event that occurred.
        """
        self.view.grid_layout.stats_layout.set_info(*event.source.parent.get_info())

    def on_use_button_callback(self: ShopScene, event: UIOnClickEvent) -> None:
        """Process use button callback logic.

        Args:
            event: The event that occurred.
        """
        self.model.registry.get_system(ShopSystem).purchase(
            self.model.player_id,
            event.source.parent.shop_index,
        )

    def on_shop_open(self: ShopScene) -> None:
        """Process shop open logic."""
        self.view.window.show_view(self)
