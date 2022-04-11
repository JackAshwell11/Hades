from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from constants.entity import (
    HEALTH_BOOST_POTION_DURATION,
    HEALTH_BOOST_POTION_INCREASE,
    StatusEffectType,
)
from constants.general import HEALTH_POTION_INCREASE
from constants.generation import TileType
from entities.base import Collectible, Item
from entities.status_effect import StatusEffect
from textures import non_moving_textures

if TYPE_CHECKING:
    import arcade
    from views.game import Game

# Get the logger
logger = logging.getLogger(__name__)


class HealthPotion(Collectible):
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

    def item_activate(self) -> bool:
        """
        Called when the item is activated by the player.

        Returns
        -------
        bool
            Whether the item activation was successful or not.
        """
        # Check if the potion can be used
        if (
            self.player.health
            == self.player.entity_type.health
            + self.player.state_modifiers["bonus health"]
        ):
            # Can't be used
            logger.info("Can't use health potion since player health is full")
            return False

        # Add health to the player
        self.player.health += HEALTH_POTION_INCREASE
        health_limit = (
            self.player.entity_type.health + self.player.state_modifiers["bonus health"]
        )
        if self.player.health > health_limit:
            self.player.health = health_limit
            logger.debug("Set player health to max")

        # Health increase successful
        self.remove_from_sprite_lists()

        # Use was successful
        logger.info("Used health potion")
        return True


class HealthBoostPotion(Collectible):
    """
    Represents a health boost potion in the game.

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
    raw_texture: arcade.Texture = non_moving_textures["items"][1]
    item_id: TileType = TileType.HEALTH_BOOST_POTION

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def __repr__(self) -> str:
        return f"<HealthBoostPotion (Position=({self.center_x}, {self.center_y}))>"

    def item_activate(self) -> bool:
        """
        Called when the item is activated by the player.

        Returns
        -------
        bool
            Whether the item activation was successful or not.
        """
        # Check if the status effect can be applied
        if StatusEffectType.HEALTH in [
            effect.effect_type for effect in self.player.applied_effects
        ]:
            logger.info("Can't use health boost potion since it is already applied")
            return False

        # Apply the health status effect
        self.player.add_status_effect(
            StatusEffect(
                self.player,
                StatusEffectType.HEALTH,
                HEALTH_BOOST_POTION_INCREASE,
                HEALTH_BOOST_POTION_DURATION,
                self.player.health,
            )
        )

        # Remove the item
        self.remove_from_sprite_lists()

        # Effect was successful
        logger.info("Used health boost potion")
        return True


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
    raw_texture: arcade.Texture = non_moving_textures["items"][2]
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
        print("shop activate")
        return False
