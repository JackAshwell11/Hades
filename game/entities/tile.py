from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.consumable import InstantEffectType
from game.entities.base import CollectibleTile, Tile, UsableTile
from game.entities.status_effect import StatusEffectBase, create_status_effect
from game.textures import non_moving_textures

if TYPE_CHECKING:
    import arcade

    from game.constants.consumable import ConsumableData
    from game.views.game_view import Game

# Get the logger
logger = logging.getLogger(__name__)


class Floor(Tile):
    """
    Represents a floor tile in the game.

    Parameters
    ----------
    x: int
        The x position of the floor tile in the game map.
    y: int
        The y position of the floor tile in the game map.
    """

    # Class variables
    raw_texture: arcade.Texture = non_moving_textures["tiles"][0]

    def __init__(
        self,
        x: int,
        y: int,
    ) -> None:
        super().__init__(x, y)

    def __repr__(self) -> str:
        return f"<Floor (Position=({self.center_x}, {self.center_y}))>"


class Wall(Tile):
    """
    Represents a wall tile in the game.

    Parameters
    ----------
    x: int
        The x position of the wall tile in the game map.
    y: int
        The y position of the wall tile in the game map.
    """

    # Class variables
    raw_texture: arcade.Texture = non_moving_textures["tiles"][1]
    blocking: bool = True

    def __init__(
        self,
        x: int,
        y: int,
    ) -> None:
        super().__init__(x, y)

    def __repr__(self) -> str:
        return f"<Wall (Position=({self.center_x}, {self.center_y}))>"


class Shop(UsableTile):
    """
    Represents a shop tile in the game.

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
    blocking: bool = True

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def __repr__(self) -> str:
        return f"<Shop (Position=({self.center_x}, {self.center_y}))>"

    def item_use(self) -> bool:
        """
        Called when the item is used by the player.

        Returns
        -------
        bool
            Whether the item activation was successful or not.
        """
        # Show the shop view and enable it's UIManager
        self.game.window.show_view(self.game.window.views["ShopView"])

        # Return true since activation will always be successful
        return True


class Consumable(UsableTile, CollectibleTile):
    """
    Represents a consumable that can be consumed by the player in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the consumable can have a reference to it.
    x: int
        The x position of the consumable in the game map.
    y: int
        The y position of the consumable in the game map.
    consumable_type: ConsumableData
        The type of this consumable.
    consumable_level: int
        The level of this consumable.
    """

    # Class variables
    item_text: str = "Press E to pick up and R to activate"

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
        consumable_type: ConsumableData,
        consumable_level: int,
    ) -> None:
        self.consumable_level: int = consumable_level
        super().__init__(game, x, y)
        self.consumable_type: ConsumableData = consumable_type
        self.texture: arcade.Texture = self.consumable_type.texture

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
        # Return the name
        return self.consumable_type.name

    def item_use(self) -> bool:
        """
        Called when the consumable is used by the player.

        Returns
        -------
        bool
            Whether the consumable use was successful or not.
        """
        # Get the adjusted level for this consumable
        adjusted_level = self.consumable_level - 1

        # Apply all the instant effects linked to this consumable
        for instant in self.consumable_type.instant:
            match instant.instant_type:
                case InstantEffectType.HEALTH:
                    if self.player.health == self.player.max_health:
                        # Can't be used
                        self.game.display_info_box("Your health is already at max")
                        logger.debug(
                            f"{self.player} health at max so instant potion"
                            "can't be used"
                        )
                        return False

                    # Add health to the player
                    self.player.health += instant.increase(adjusted_level)
                    if self.player.health > self.player.max_health:
                        self.player.health = self.player.max_health
                        logger.debug("Set player health to max")
                case InstantEffectType.ARMOUR:
                    if self.player.armour == self.player.max_armour:
                        # Can't be used
                        self.game.display_info_box("Your armour is already at max")
                        logger.debug(
                            f"{self.player} armour at max so instant potion"
                            "can't be used"
                        )
                        return False

                    # Add armour to the player
                    self.player.armour += instant.increase(adjusted_level)
                    if self.player.armour > self.player.max_armour:
                        self.player.armour = self.player.max_armour
                        logger.debug("Set player armour to max")

        # Apply all the status effects linked to this consumable
        for effect in self.consumable_type.status_effects:
            # Check if the status effect can be applied
            if effect.status_type in [
                player_effect.status_effect_type
                for player_effect in self.player.applied_effects
            ]:
                self.game.display_info_box(
                    f"A {effect.status_type.value} status effect is already applied"
                )
                logger.debug(f"{effect.status_type} already applied to player")
                return False

            # Apply the status effect
            new_effect: StatusEffectBase = create_status_effect(
                effect.status_type,
                self.player,
                effect.increase(adjusted_level),
                effect.duration(adjusted_level),
            )
            self.player.applied_effects.append(new_effect)
            new_effect.apply_effect()

        # Remove the item
        self.remove_from_sprite_lists()

        # Effect was successful
        logger.info(f"Used {self.consumable_type.name} potion")
        return True
