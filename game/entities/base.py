from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants.entity import ARMOUR_REGEN_AMOUNT, ARMOUR_REGEN_WAIT
from constants.enums import EntityID, TileType
from constants.general import SPRITE_SCALE
from entities.attack import AttackBase
from textures import pos_to_pixel

if TYPE_CHECKING:
    from constants.entity import BaseType, EnemyType, EntityType, PlayerType
    from entities.player import Player
    from views.game import Game

# Get the logger
logger = logging.getLogger(__name__)


class Entity(arcade.Sprite):
    """
    Represents an entity in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the entity can have a reference to it.
    x: int
        The x position of the entity in the game map.
    y: int
        The y position of the entity in the game map.
    entity_type: BaseType
        The constant data about this specific entity. This allows me to support multiple
        player and enemy types later on.

    Attributes
    ----------
    health: int
        The health of this entity.
    armour: int
        The armour of this entity.
    direction: float
        The angle the entity is facing.
    facing: int
        The direction the entity is facing. 0 is right and 1 is left.
    time_since_last_attack: float
        The time since the last attack.
    time_out_of_combat: float
        The time since the entity was last in combat.
    time_since_armour_regen: float
        The time since the entity last regenerated armour.
    """

    # Class variables
    ID: EntityID = EntityID.ENTITY

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
        entity_type: BaseType,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.game: Game = game
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.entity_data: BaseType = entity_type
        self.texture: arcade.Texture = self.entity_type.textures["idle"][0][0]
        self.health: int = self.entity_type.health
        self.armour: int = self.entity_type.armour
        self.attack_algorithms: list[AttackBase] = [
            algorithm.value(self) for algorithm in self.entity_type.attack_algorithms
        ]
        self.direction: float = 0
        self.facing: int = 0
        self.time_since_last_attack: float = 0
        self.time_out_of_combat: float = 0
        self.time_since_armour_regen: float = self.entity_type.armour_regen_cooldown

    def __repr__(self) -> str:
        return f"<Entity (Position=({self.center_x}, {self.center_y}))>"

    @property
    def entity_type(self) -> EntityType:
        """Returns the general entity data."""
        return self.entity_data.entity_type

    @property
    def custom_data(self) -> PlayerType | EnemyType:
        """Returns the specific data about this entity"""
        return self.entity_data.custom_data

    def on_update(self, delta_time: float = 1 / 60) -> None:
        """
        Processes movement and game logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        """
        raise NotImplementedError

    def deal_damage(self, damage: int) -> None:
        """
        Deals damage to an entity.

        Parameters
        ----------
        damage: int
            The amount of health to take away from the entity.
        """
        # Check if the entity still has armour
        if self.armour > 0:
            # Damage the armour
            self.armour -= damage
            if self.armour < 0:
                # Damage exceeds armour so damage health
                self.health += self.armour
                self.armour = 0
        else:
            # Damage the health
            self.health -= damage
        logger.debug(f"Dealing {damage} to {self}")

        # Check if the entity should be killed
        if self.health <= 0:
            self.remove_from_sprite_lists()
            logger.info(f"Killed {self}")

    def check_armour_regen(self, delta_time: float) -> None:
        """
        Checks if the entity can regenerate armour.

        Parameters
        ----------
        delta_time:
            Time interval since the last time the function was called.
        """
        # Check if the entity has been out of combat for ARMOUR_REGEN_WAIT seconds
        if self.time_out_of_combat >= ARMOUR_REGEN_WAIT:
            # Check if enough has passed since the last armour regen
            if self.time_since_armour_regen >= self.entity_type.armour_regen_cooldown:
                # Regen armour
                self.armour += ARMOUR_REGEN_AMOUNT
                self.time_since_armour_regen = 0
                logger.debug(f"Regenerated armour for {self}")
            else:
                # Increment the counter since not enough time has passed
                self.time_since_armour_regen += delta_time
        else:
            # Increment the counter since not enough time has passed
            self.time_out_of_combat += delta_time


class Tile(arcade.Sprite):
    """
    Represents a tile in the game.

    Parameters
    ----------
    x: int
        The x position of the tile in the game map.
    y: int
        The y position of the tile in the game map.

    Attributes
    ----------
    center_x: float
        The x position of the tile on the screen.
    center_y: float
        The y position of the tile on the screen.
    texture: arcade.Texture
        The sprite which represents this tile.
    """

    # Class variables
    raw_texture: arcade.Texture | None = None
    is_static: bool = False

    def __init__(
        self,
        x: int,
        y: int,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.texture: arcade.Texture = self.raw_texture

    def __repr__(self) -> str:
        return f"<Tile (Position=({self.center_x}, {self.center_y}))>"


class Item(Tile):
    """
    Represents an item in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the item can have a reference to it.
    x: int
        The x position of the item in the game map.
    y: int
        The y position of the item in the game map.
    """

    # Class variables
    item_id: TileType = TileType.NONE
    item_text: str = "Press E to activate"

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(x, y)
        self.game: Game = game

    def __repr__(self) -> str:
        return f"<Item (Position=({self.center_x}, {self.center_y}))>"

    @property
    def player(self) -> Player:
        """Returns the player object for ease of access."""
        return self.game.player

    def item_activate(self) -> bool:
        """
        Called when the item is activated by the player. Override this to add item
        activate functionality.

        Returns
        -------
        bool
            Whether the item activation was successful or not.
        """
        return False


class Collectible(Item):
    """
    Represents a collectible item in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the item can have a reference to it.
    x: int
        The x position of the item in the game map.
    y: int
        The y position of the item in the game map.
    """

    # Class variables
    item_text: str = "Press E to pick up and R to activate"

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)

    def item_pick_up(self) -> bool:
        """
        Called when the item is picked up by the player.

        Returns
        -------
        bool
            Whether the item pickup was successful or not.
        """
        # Try and add the item to the player's inventory
        if self.player.add_item_to_inventory(self):
            # Add successful
            self.remove_from_sprite_lists()

            # Activate was successful
            logger.info(f"Picked up item {self}")
            return True
        else:
            # Add not successful. TO DO: Probably give message to user
            logger.info(f"Can't pick up item {self}")
            return False
