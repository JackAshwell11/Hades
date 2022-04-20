from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.consumable import (
    ARMOUR_BOOST_POTION,
    ARMOUR_BOOST_POTION_DURATION,
    ARMOUR_BOOST_POTION_INCREASE,
    ARMOUR_POTION,
    ARMOUR_POTION_INCREASE,
    FIRE_RATE_BOOST_POTION,
    FIRE_RATE_BOOST_POTION_DURATION,
    FIRE_RATE_BOOST_POTION_INCREASE,
    HEALTH_BOOST_POTION,
    HEALTH_BOOST_POTION_DURATION,
    HEALTH_BOOST_POTION_INCREASE,
    HEALTH_POTION,
    HEALTH_POTION_INCREASE,
    SPEED_BOOST_POTION,
    SPEED_BOOST_POTION_DURATION,
    SPEED_BOOST_POTION_INCREASE,
    StatusEffectType,
)
from game.constants.generation import TileType
from game.entities.base import Collectible, Item
from game.entities.status_effect import StatusEffect
from game.textures import non_moving_textures

if TYPE_CHECKING:
    import arcade

    from game.constants.consumable import ConsumableData
    from game.views.game import Game

# Get the logger
logger = logging.getLogger(__name__)


class HealthPotion(Collectible):
    """
    Represents a health potion in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the health potion can have a reference to it.
    x: int
        The x position of the health potion in the game map.
    y: int
        The y position of the health potion in the game map.
    """

    # Class variables
    item_id: TileType = TileType.HEALTH_POTION
    consumable_type: ConsumableData = HEALTH_POTION

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
        Called when the health potion is activated by the player.

        Returns
        -------
        bool
            Whether the health potion activation was successful or not.
        """
        # Check if the potion can be used
        if self.player.health == self.player.max_health:
            # Can't be used
            logger.info("Can't use health potion since player health is full")
            return False

        # Add health to the player
        self.player.health += HEALTH_POTION_INCREASE
        if self.player.health > self.player.max_health:
            self.player.health = self.player.max_health
            logger.debug("Set player health to max")

        # Health increase successful
        self.remove_from_sprite_lists()

        # Use was successful
        logger.info("Used health potion")
        return True


class ArmourPotion(Collectible):
    """
    Represents an armour potion in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the armour potion can have a reference to it.
    x: int
        The x position of the armour potion in the game map.
    y: int
        The y position of the armour potion in the game map.
    """

    # Class variables
    item_id: TileType = TileType.ARMOUR_POTION
    consumable_type: ConsumableData = ARMOUR_POTION

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def __repr__(self) -> str:
        return f"<ArmourPotion (Position=({self.center_x}, {self.center_y}))>"

    def item_activate(self) -> bool:
        """
        Called when the armour potion is activated by the player.

        Returns
        -------
        bool
            Whether the armour potion activation was successful or not.
        """
        # Check if the potion can be used
        if self.player.armour == self.player.max_armour:
            # Can't be used
            logger.info("Can't use armour potion since player armour is full")
            return False

        # Add armour to the player
        self.player.armour += ARMOUR_POTION_INCREASE
        if self.player.armour > self.player.max_armour:
            self.player.armour = self.player.max_armour
            logger.debug("Set player armour to max")

        # Armour increase successful
        self.remove_from_sprite_lists()

        # Use was successful
        logger.info("Used armour potion")
        return True


class HealthBoostPotion(Collectible):
    """
    Represents a health boost potion in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the health boost potion can have a reference to
        it.
    x: int
        The x position of the health boost potion in the game map.
    y: int
        The y position of the health boost potion in the game map.
    """

    # Class variables
    item_id: TileType = TileType.HEALTH_BOOST_POTION
    consumable_type: ConsumableData = HEALTH_BOOST_POTION

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
        Called when the health boost potion is activated by the player.

        Returns
        -------
        bool
            Whether the health boost potion activation was successful or not.
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


class ArmourBoostPotion(Collectible):
    """
    Represents an armour boost potion in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the armour boost potion can have a reference to
        it.
    x: int
        The x position of the armour boost potion in the game map.
    y: int
        The y position of the armour boost potion in the game map.
    """

    # Class variables
    item_id: TileType = TileType.ARMOUR_BOOST_POTION
    consumable_type: ConsumableData = ARMOUR_BOOST_POTION

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def __repr__(self) -> str:
        return f"<ArmourBoostPotion (Position=({self.center_x}, {self.center_y}))>"

    def item_activate(self) -> bool:
        """
        Called when the armour boost potion is activated by the player.

        Returns
        -------
        bool
            Whether the armour boost potion activation was successful or not.
        """
        # Check if the status effect can be applied
        if StatusEffectType.ARMOUR in [
            effect.effect_type for effect in self.player.applied_effects
        ]:
            logger.info("Can't use armour boost potion since it is already applied")
            return False

        # Apply the armour status effect
        self.player.add_status_effect(
            StatusEffect(
                self.player,
                StatusEffectType.ARMOUR,
                ARMOUR_BOOST_POTION_INCREASE,
                ARMOUR_BOOST_POTION_DURATION,
                self.player.armour,
            )
        )

        # Remove the item
        self.remove_from_sprite_lists()

        # Effect was successful
        logger.info("Used armour boost potion")
        return True


class SpeedBoostPotion(Collectible):
    """
    Represents a speed potion in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the speed boost potion can have a reference to
        it.
    x: int
        The x position of the speed boost potion in the game map.
    y: int
        The y position of the speed boost potion in the game map.
    """

    # Class variables
    item_id: TileType = TileType.SPEED_BOOST_POTION
    consumable_type: ConsumableData = SPEED_BOOST_POTION

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def __repr__(self) -> str:
        return f"<SpeedBoostPotion (Position=({self.center_x}, {self.center_y}))>"

    def item_activate(self) -> bool:
        """
        Called when the speed boost potion is activated by the player.

        Returns
        -------
        bool
            Whether the speed boost potion activation was successful or not.
        """
        # Check if the status effect can be applied
        if StatusEffectType.SPEED in [
            effect.effect_type for effect in self.player.applied_effects
        ]:
            logger.info("Can't use speed boost potion since it is already applied")
            return False

        # Apply the speed status effect
        self.player.add_status_effect(
            StatusEffect(
                self.player,
                StatusEffectType.SPEED,
                SPEED_BOOST_POTION_INCREASE,
                SPEED_BOOST_POTION_DURATION,
                self.player.max_velocity,
            )
        )

        # Remove the item
        self.remove_from_sprite_lists()

        # Effect was successful
        logger.info("Used speed boost potion")
        return True


class FireRateBoostPotion(Collectible):
    """
    Represents a fire rate boost potion in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the fire rate boost potion can have a reference
        to it.
    x: int
        The x position of the fire rate boost potion in the game map.
    y: int
        The y position of the fire rate boost potion in the game map.
    """

    # Class variables
    item_id: TileType = TileType.FIRE_RATE_BOOST_POTION
    consumable_type: ConsumableData = FIRE_RATE_BOOST_POTION

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def __repr__(self) -> str:
        return f"<FireRateBoostPotion (Position=({self.center_x}, {self.center_y}))>"

    def item_activate(self) -> bool:
        """
        Called when the fire rate boost potion is activated by the player.

        Returns
        -------
        bool
            Whether the fire rate boost potion activation was successful or not.
        """
        # Check if the status effect can be applied
        if StatusEffectType.FIRE_RATE in [
            effect.effect_type for effect in self.player.applied_effects
        ]:
            logger.info("Can't use fire rate boost potion since it is already applied")
            return False

        # Apply the fire rate status effect
        self.player.add_status_effect(
            StatusEffect(
                self.player,
                StatusEffectType.FIRE_RATE,
                FIRE_RATE_BOOST_POTION_INCREASE,
                FIRE_RATE_BOOST_POTION_DURATION,
                self.player.bonus_attack_cooldown,
            )
        )

        # Remove the item
        self.remove_from_sprite_lists()

        # Effect was successful
        logger.info("Used fire rate boost potion")
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
