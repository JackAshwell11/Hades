from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.generation import TileType
from game.entities.base import Item
from game.textures import non_moving_textures

if TYPE_CHECKING:
    import arcade

    from game.views.game import Game

# Get the logger
logger = logging.getLogger(__name__)


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
    raw_texture: arcade.Texture = non_moving_textures["items"][6]
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

    def item_activate(self) -> bool:
        """
        Called when the item is activated by the player.

        Returns
        -------
        bool
            Whether the item activation was successful or not.
        """
        # Show the shop view and enable it's UIManager
        self.game.window.show_view(self.game.window.views["ShopView"])
        self.game.window.views["ShopView"].manager.enable()

        # Return true since activation will always be successful
        return True
