"""Stores the different static tiles that can exist in the game."""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from hades.constants.game_object import ConsumableData, InstantEffectType
from hades.game_objects.base import CollectibleTile, Tile, UsableTile
from hades.textures import non_moving_textures

if TYPE_CHECKING:
    import arcade

    from hades.views.game_view import Game

__all__ = (
    "Consumable",
    "Floor",
    "Wall",
)

# Get the logger
logger = logging.getLogger(__name__)


class Floor(Tile):
    """Represents a floor tile in the game."""

    # Class variables
    raw_texture: arcade.Texture = non_moving_textures["tiles"][0]

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<Floor (Position=({self.center_x}, {self.center_y}))>"


class Wall(Tile):
    """Represents a wall tile in the game."""

    # Class variables
    raw_texture: arcade.Texture = non_moving_textures["tiles"][1]
    blocking: bool = True

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<Wall (Position=({self.center_x}, {self.center_y}))>"


# class Shop(UsableTile):
#     """Represents a shop tile in the game."""
#
#     # Class variables
#     raw_texture: arcade.Texture = non_moving_textures["items"][6]
#     blocking: bool = True
#
#     def __repr__(self) -> str:
#         return f"<Shop (Position=({self.center_x}, {self.center_y}))>"
#
#     def item_use(self) -> bool:
#         """
#         Called when the item is used by the player.
#
#         Returns
#         -------
#         bool
#             Whether the item activation was successful or not.
#         """
#         # Show the shop view and enable it's UIManager
#         self.game.window.show_view(self.game.window.views["ShopView"])
#
#         # Return true since activation will always be successful
#         return True


class Consumable(UsableTile, CollectibleTile):
    """Represents a consumable that can be consumed by the player in the game.

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
        consumable_level: int = 0,
    ) -> None:
        self.consumable_level: int = consumable_level
        super().__init__(game, x, y)
        self.consumable_type: ConsumableData = consumable_type
        self.texture: arcade.Texture = self.consumable_type.texture

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<Consumable (Position=({self.center_x}, {self.center_y}))>"

    @property
    def name(self) -> str:
        """Get the name of this consumable.

        Returns
        -------
        str
            The name of this consumable.
        """
        # Return the name
        return self.consumable_type.name

    def item_use(self) -> bool:
        """Process item use functionality.

        Returns
        -------
        bool
            Whether the consumable use was successful or not.
        """
        # Get the adjusted level for this consumable
        adjusted_level = self.consumable_level - 1

        # Apply all the instant effects linked to this consumable
        for instant in self.consumable_type.instant:
            if instant.instant_type is InstantEffectType.HEALTH:
                if self.player.health.value == self.player.health.max_value:
                    # Can't be used
                    self.game.display_info_box("Your health is already at max")
                    logger.debug(
                        "%r health at max so instant potion can't be used",
                        self.player,
                    )
                    return False

                # Add health to the player
                self.player.health.value = self.player.health.value + instant.increase(
                    adjusted_level
                )
                if self.player.health.value > self.player.health.max_value:
                    self.player.health.value = self.player.health.max_value
                    logger.debug("Set player health to max")
            elif instant.instant_type is InstantEffectType.ARMOUR:
                if self.player.armour.value == self.player.armour.max_value:
                    # Can't be used
                    self.game.display_info_box("Your armour is already at max")
                    logger.debug(
                        "%r armour at max so instant potion can't be used",
                        self.player,
                    )
                    return False

                # Add armour to the player
                self.player.armour.value = self.player.armour.value + instant.increase(
                    adjusted_level
                )
                if self.player.armour.value > self.player.armour.max_value:
                    self.player.armour.value = self.player.armour.max_value
                    logger.debug("Set player armour to max")

        # Apply all the status effects linked to this consumable
        for effect in self.consumable_type.status_effects:
            # Check if the status effect can be applied
            if effect.status_type in [
                getattr(attribute.applied_status_effect, "status_effect_type", None)
                for attribute in self.player.entity_state.values()
            ]:
                self.game.display_info_box(
                    f"A {effect.status_type.value} status effect is already applied"
                )
                logger.debug("%r already applied to player", effect.status_type)
                return False

            # Apply the status effect
            self.player.entity_state[effect.status_type.value].apply_status_effect(
                effect, adjusted_level
            )

        # Remove the item
        self.remove_from_sprite_lists()

        # Effect was successful
        logger.info("Used %r potion", self.consumable_type.name)
        return True
