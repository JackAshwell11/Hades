from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.consumable import (
    ARMOUR_BOOST_POTION,
    ARMOUR_POTION,
    FIRE_RATE_BOOST_POTION,
    HEALTH_BOOST_POTION,
    HEALTH_POTION,
    SPEED_BOOST_POTION,
    ConsumableLevelData,
    InstantEffectType,
)
from game.constants.generation import TileType
from game.entities.base import Item
from game.entities.status_effect import StatusEffect
from game.textures import non_moving_textures

if TYPE_CHECKING:
    import arcade

    from game.constants.consumable import ConsumableData
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


class Consumable(Item):
    """
    Represents a consumable in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the consumable can have a reference to it.
    x: int
        The x position of the consumable in the game map.
    y: int
        The y position of the consumable in the game map.
    """

    # Class variables
    item_text: str = "Press E to pick up and R to activate"
    consumable_type: ConsumableData | None = None
    consumable_level: int = 1

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)
        if self.consumable_type is not None:
            self.texture: arcade.Texture = self.consumable_type.levels[
                self.consumable_level
            ].texture
        else:
            raise ValueError(f"Consumable type for {self} can't be None")

    def __repr__(self) -> str:
        return f"<Consumable (Position=({self.center_x}, {self.center_y}))>"

    @property
    def name(self) -> str:
        """
        Gets the name of this consumable.

        Returns
        -------
        str
            The name of this consumable.
        """
        # Make sure the consumable type is valid
        assert self.consumable_type is not None

        # Return the name
        return self.consumable_type.name

    def get_level(self, level: int) -> ConsumableLevelData | None:
        """
        Gets a specific level for this consumable.

        Parameters
        ----------
        level: int
            The level to get data for.

        Returns
        -------
        ConsumableLevelData | None
            The level data if it exists.
        """
        # Make sure the consumable type is valid
        assert self.consumable_type is not None

        # Return the level if it exists
        return self.consumable_type.levels.get(level, None)

    def item_activate(self) -> bool:
        """
        Called when the health boost potion is activated by the player.

        Returns
        -------
        bool
            Whether the health boost potion activation was successful or not.
        """
        # Make sure the consumable type is valid
        assert self.consumable_type is not None

        # Apply all the instant effects linked to this consumable
        for instant in self.consumable_type.levels[self.consumable_level].instant:
            match instant.instant_type:
                case InstantEffectType.HEALTH:
                    if self.player.health == self.player.max_health:
                        # Can't be used
                        return False

                    # Add health to the player
                    self.player.health += instant.value
                    if self.player.health > self.player.max_health:
                        self.player.health = self.player.max_health
                        logger.debug("Set player health to max")
                case InstantEffectType.ARMOUR:
                    if self.player.armour == self.player.max_armour:
                        # Can't be used
                        return False

                    # Add armour to the player
                    self.player.armour += instant.value
                    if self.player.armour > self.player.max_armour:
                        self.player.armour = self.player.max_armour
                        logger.debug("Set player armour to max")

        # Apply all the status effects linked to this consumable
        for effect in self.consumable_type.levels[self.consumable_level].status_effects:
            # Check if the status effect can be applied
            if effect.status_type in [
                player_effect.effect_type
                for player_effect in self.player.applied_effects
            ]:
                return False

            # Apply the status effect
            new_effect = StatusEffect(
                self.player,
                effect.status_type,
                effect.value,
                effect.duration,
            )
            self.player.applied_effects.append(new_effect)
            new_effect.apply_effect()

        # Remove the item
        self.remove_from_sprite_lists()

        # Effect was successful
        logger.info(f"Used {self.consumable_type.name} potion")
        return True

    def item_pick_up(self) -> bool:
        """
        Called when the consumable is picked up by the player.

        Returns
        -------
        bool
            Whether the consumable pickup was successful or not.
        """
        # Try and add the item to the player's inventory
        if self.player.add_item_to_inventory(self):
            # Add successful
            self.remove_from_sprite_lists()

            # Activate was successful
            logger.info(f"Picked up consumable {self}")
            return True
        else:
            # Add not successful. TODO: Probably give message to user
            logger.info(f"Can't pick up consumable {self}")
            return False


class HealthPotion(Consumable):
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


class ArmourPotion(Consumable):
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


class HealthBoostPotion(Consumable):
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


class ArmourBoostPotion(Consumable):
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


class SpeedBoostPotion(Consumable):
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


class FireRateBoostPotion(Consumable):
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
