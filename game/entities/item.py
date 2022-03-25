from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants import TileType
from entities.base import Item
from textures import non_moving_textures

if TYPE_CHECKING:
    from views.game import Game


class HealthPotion(Item):
    """
    Represents a health potion in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the item can have a reference to it.
    x: int
        The x position of the health potion in the game map.
    y: int
        The y position of the health potion in the game map.
    """

    # Class variables
    raw_texture: arcade.Texture = non_moving_textures["items"][0]
    item_id: TileType = TileType.HEALTH_POTION

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def __repr__(self) -> str:
        return f"<HealthPotion (Position=({self.center_x}, {self.center_y}))>"

    def item_activate(self) -> None:
        """Called when the item is activated by the player."""
        try:
            # Try and add the item to the player's inventory
            self.player.inventory_obj.add_item(self)

            # Add successful
            self.remove_from_sprite_lists()
        except IndexError:
            # Add not successful. TO DO: Probably give message to user
            pass


class Shop(Item):
    """
    Represents a shop item in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the item can have a reference to it.
    x: int
        The x position of the shop item in the game map.
    y: int
        The y position of the shop item in the game map.
    """

    # Class variables
    raw_texture: arcade.Texture = non_moving_textures["items"][1]
    is_static: bool = True
    item_id: TileType = TileType.HEALTH_POTION

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def __repr__(self) -> str:
        return f"<Shop (Position=({self.center_x}, {self.center_y}))>"

    def item_activate(self) -> None:
        """Called when the item is activated by the player."""
        print("shop")
