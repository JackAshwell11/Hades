from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from game.constants.entity import (
    ARMOUR_REGEN_AMOUNT,
    ARMOUR_REGEN_WAIT,
    SPRITE_SCALE,
    EntityID,
    EntityUpgradeData,
)
from game.constants.generation import TileType
from game.textures import pos_to_pixel

if TYPE_CHECKING:
    from game.constants.entity import (
        AreaOfEffectAttackData,
        AttackData,
        BaseData,
        EnemyData,
        EntityData,
        MeleeAttackData,
        PlayerData,
        RangedAttackData,
    )
    from game.entities.attack import AttackBase
    from game.entities.player import Player
    from game.entities.status_effect import StatusEffect
    from game.views.game import Game

# Get the logger
logger = logging.getLogger(__name__)


class IndicatorBar:
    """
    Represents a bar which can display information about an entity.

    Parameters
    ----------
    owner: Entity
        The owner of this indicator bar.
    position: tuple[float, float]
        The initial position of the bar.
    full_color: arcade.Color
        The color of the bar.
    background_color: arcade.Color
        The background color of the bar.
    width: int
        The width of the bar.
    height: int
        The height of the bar.
    border_size: int
        The size of the bar's border.
    """

    __slots__ = (
        "owner",
        "_bar_width",
        "_bar_height",
        "_half_bar_width",
        "_half_bar_height",
        "_center_x",
        "_center_y",
        "_fullness",
        "_background_box",
        "_full_box",
    )

    def __init__(
        self,
        owner: Entity,
        position: tuple[float, float] = (0, 0),
        full_color: arcade.Color = arcade.color.GREEN,
        background_color: arcade.Color = arcade.color.BLACK,
        width: int = 50,
        height: int = 4,
        border_size: int = 4,
    ) -> None:
        # Store the reference to the owner
        self.owner: Entity = owner

        # Set the needed size variables
        self._bar_width: int = width
        self._bar_height: int = height
        self._half_bar_width: float = self.bar_width / 2
        self._half_bar_height: float = self.bar_height / 2
        self._center_x: float = 0.0
        self._center_y: float = 0.0
        self._fullness: float = 0.0

        # Create the boxes needed to represent the indicator bar
        self._background_box: arcade.SpriteSolidColor = arcade.SpriteSolidColor(
            self.bar_width + border_size,
            self.bar_height + border_size,
            background_color,
        )
        self._full_box: arcade.SpriteSolidColor = arcade.SpriteSolidColor(
            self.bar_width,
            self.bar_height,
            full_color,
        )
        self.owner.game.indicator_bar_sprites.append(self.background_box)
        self.owner.game.indicator_bar_sprites.append(self.full_box)

        # Set the fullness and position of the bar
        self.fullness: float = 1.0
        self.position: tuple[float, float] = position

    def __repr__(self) -> str:
        return f"<IndicatorBar (Owner={self.owner})>"

    @property
    def background_box(self) -> arcade.SpriteSolidColor:
        """
        Gets the background box object of the indicator bar.

        Returns
        -------
        arcade.SpriteSolidColor
            The background box object.
        """
        return self._background_box

    @property
    def full_box(self) -> arcade.SpriteSolidColor:
        """
        Gets the full box of the indicator bar.

        Returns
        -------
        arcade.SpriteSolidColor
            The full box object.
        """
        return self._full_box

    @property
    def bar_width(self) -> int:
        """
        Gets the width of the bar.

        Returns
        -------
        int
            The width of the bar.
        """
        return self._bar_width

    @property
    def half_bar_width(self) -> float:
        """
        Gets the value that is half of the bar's width.

        Returns
        -------
        float
            The half bar width.
        """
        return self._half_bar_width

    @property
    def bar_height(self) -> int:
        """
        Gets the height of the bar.

        Returns
        -------
        int
            The height of the bar.
        """
        return self._bar_height

    @property
    def half_bar_height(self) -> float:
        """
        Gets the value that is half of the bar's height.

        Returns
        -------
        float
            The half bar height.
        """
        return self._half_bar_height

    @property
    def center_x(self) -> float:
        """
        Gets the x position of the bar.

        Returns
        -------
        float
            The x position of the bar.
        """
        return self._center_x

    @property
    def center_y(self) -> float:
        """
        Gets the y position of the bar.

        Returns
        -------
        float
            The y position of the bar.
        """
        return self._center_y

    @property
    def fullness(self) -> float:
        """
        Gets the fullness of the bar.

        Returns
        -------
        float
            The fullness of the bar.
        """
        return self._fullness

    @fullness.setter
    def fullness(self, new_fullness: float) -> None:
        """
        Sets the fullness of the bar.

        Parameters
        ----------
        new_fullness: float
            The new fullness of the bar

        Raises
        ------
        ValueError
            The fullness must be between 0.0 and 1.0.
        """
        # Check if new_fullness if valid
        if not (0.0 <= new_fullness <= 1.0):
            raise ValueError(
                f"Got {new_fullness}, but fullness must be between 0.0 and 1.0."
            )

        # Set the size of the bar
        self._fullness = new_fullness
        if new_fullness == 0.0:
            # Set the full_box to not be visible since it is not full anymore
            self.full_box.visible = False
        else:
            # Set the full_box to be visible incase it wasn't then update the bar
            self.full_box.visible = True
            self.full_box.width = self.bar_width * new_fullness
            self.full_box.left = self.center_x - (self.bar_width // 2)

    @property
    def position(self) -> tuple[float, float]:
        """
        Gets the current position of the bar.

        Returns
        -------
        tuple[float, float]
            The current position of the bar.
        """
        return self.center_x, self.center_y

    @position.setter
    def position(self, new_position: tuple[float, float]) -> None:
        """
        Sets the new position of the bar.

        Parameters
        ----------
        new_position: tuple[float, float]
            The new position of the bar.
        """
        # Check if the position has changed. If so, change the bar's position
        if new_position != self.position:
            self._center_x, self._center_y = new_position
            self.background_box.position = new_position
            self.full_box.position = new_position

            # Make sure full_box is to the left of the bar instead of the middle
            self.full_box.left = self.center_x - (self.bar_width // 2)


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
    entity_type: BaseData
        The raw data for this entity.

    Attributes
    ----------
    attack_algorithms: list[AttackBase]
        A list of the entity's attack algorithms.
    health_bar: IndicatorBar
        An indicator bar object which displays the entity's health visually.
    armour_bar: IndicatorBar
        An indicator bar object which displays the entity's armour visually.
    applied_effects: list[StatusEffect]
        The currently applied status effects.
    current_attack_index: int
        The index of the currently selected attack.
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
    entity_id: EntityID = EntityID.ENTITY

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
        entity_type: BaseData,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.game: Game = game
        self.center_x, self.center_y = pos_to_pixel(x, y)
        self.entity_type: BaseData = entity_type
        self.texture: arcade.Texture = self.entity_data.textures["idle"][0][0]
        self.attack_algorithms: list[AttackBase] = [
            algorithm.attack_type.value(self, algorithm.attack_cooldown)
            for algorithm in self.attacks
        ]
        self.health_bar: IndicatorBar = IndicatorBar(self, (0, 0), arcade.color.RED)
        self.armour_bar: IndicatorBar = IndicatorBar(
            self,
            (0, 0),
            arcade.color.SILVER,
        )
        self._entity_state: dict[str, float] = self._initialise_entity_state()
        self.applied_effects: list[StatusEffect] = []
        self.current_attack_index: int = 0
        self.direction: float = 0
        self.facing: int = 0
        self.time_since_last_attack: float = 0
        self.time_out_of_combat: float = 0
        self.time_since_armour_regen: float = self.armour_regen_cooldown

    def __repr__(self) -> str:
        return f"<Entity (Position=({self.center_x}, {self.center_y}))>"

    @property
    def entity_data(self) -> EntityData:
        """
        Gets the general entity data.

        Returns
        -------
        EntityData
            The general entity data.
        """
        # Make sure the entity type is valid
        assert self.entity_type is not None

        # Return the entity data
        return self.entity_type.entity_data

    @property
    def player_data(self) -> PlayerData:
        """
        Gets the player data if it exists.

        Returns
        -------
        PlayerData
            The player data.
        """
        # Make sure the entity type is valid
        assert self.entity_type is not None
        assert self.entity_type.player_data is not None

        # Return the player data
        return self.entity_type.player_data

    @property
    def enemy_data(self) -> EnemyData:
        """
        Gets the enemy data if it exists.

        Returns
        -------
        EnemyData
            The enemy data.
        """
        # Make sure the entity type is valid
        assert self.entity_type is not None
        assert self.entity_type.enemy_data is not None

        # Return the enemy data
        return self.entity_type.enemy_data

    @property
    def attacks(self) -> list[AttackData]:
        """
        Gets all the attacks the entity has.

        Returns
        -------
        list[AttackData]
            The entity's attacks.
        """
        # Make sure the entity type is valid
        assert self.entity_type is not None

        # Return the enemy data
        return self.entity_type.get_all_attacks()

    @property
    def ranged_attack_data(self) -> RangedAttackData:
        """
        Gets the ranged attack data if the entity has the attack.

        Returns
        -------
        RangedAttackData
            The ranged attack data.
        """
        # Make sure the entity type is valid
        assert self.entity_type is not None
        assert self.entity_type.ranged_attack_data is not None

        # Return the ranged attack data
        return self.entity_type.ranged_attack_data

    @property
    def melee_attack_data(self) -> MeleeAttackData:
        """
        Gets the melee attack data if the entity has the attack.

        Returns
        -------
        MeleeAttackData
            The melee attack data.
        """
        # Make sure the entity type is valid
        assert self.entity_type is not None
        assert self.entity_type.melee_attack_data is not None

        # Return the melee attack data
        return self.entity_type.melee_attack_data

    @property
    def area_of_effect_attack_data(self) -> AreaOfEffectAttackData:
        """
        Gets the area of effect attack data if the entity has the attack.

        Returns
        -------
        AreaOfEffectAttackData
            The area of effect attack data.
        """
        # Make sure the entity type is valid
        assert self.entity_type is not None
        assert self.entity_type.area_of_effect_attack_data is not None

        # Return the area of effect attack data
        return self.entity_type.area_of_effect_attack_data

    @property
    def current_attack(self) -> AttackBase:
        """
        Gets the currently selected attack algorithm.

        Returns
        -------
        AttackBase
            The currently selected attack algorithm.
        """
        return self.attack_algorithms[self.current_attack_index]

    @property
    def upgrade_data(self) -> list[EntityUpgradeData]:
        """
        Gets the upgrades that are available to the entity.

        Returns
        -------
        list[game.constants.entity.EntityUpgradeData]
            The upgrades that are available to the entity.
        """
        return self.entity_data.upgrade_data

    @property
    def health(self) -> float:
        """
        Gets the entity's health.

        Returns
        -------
        float
            The entity's health
        """
        return self._entity_state["health"]

    @health.setter
    def health(self, value: float) -> None:
        """
        Sets the entity's health.

        Parameters
        ----------
        value: float
            The new health value.
        """
        self._entity_state["health"] = value

    @property
    def max_health(self) -> float:
        """
        Gets the player's maximum health.

        Returns
        -------
        float
            The player's maximum health.
        """
        return self._entity_state["max health"]

    @max_health.setter
    def max_health(self, value: float) -> None:
        """
        Sets the player's maximum health.


        Parameters
        ----------
        value: float
            The new maximum health value.
        """
        self._entity_state["max health"] = value

    @property
    def armour(self) -> float:
        """
        Gets the entity's armour.

        Returns
        -------
        float
            The entity's armour.
        """
        return self._entity_state["armour"]

    @armour.setter
    def armour(self, value: float) -> None:
        """
        Sets the entity's armour.

        Parameters
        ----------
        value: float
            The new armour value.
        """
        self._entity_state["armour"] = value

    @property
    def max_armour(self) -> float:
        """
        Gets the player's maximum armour.

        Returns
        -------
        float
            The player's maximum armour
        """
        return self._entity_state["max armour"]

    @max_armour.setter
    def max_armour(self, value: float) -> None:
        """
        Sets the player's maximum armour.

        Parameters
        ----------
        value: float
            The new maximum armour value.
        """
        self._entity_state["max armour"] = value

    @property
    def max_velocity(self) -> float:
        """
        Gets the entity's max velocity.

        Returns
        -------
        float
            The entity's max velocity.
        """
        return self._entity_state["max velocity"]

    @max_velocity.setter
    def max_velocity(self, value: float) -> None:
        """
        Sets the entity's max velocity.

        Parameters
        ----------
        value: float
            The new max velocity value.
        """
        self._entity_state["max velocity"] = value
        self.pymunk.max_velocity = value

    @property
    def armour_regen_cooldown(self) -> float:
        """
        Gets the entity's armour regen cooldown

        Returns
        -------
        float
            The entity's armour regen cooldown.
        """
        return self._entity_state["armour regen cooldown"]

    @armour_regen_cooldown.setter
    def armour_regen_cooldown(self, value: float) -> None:
        """
        Sets the entity's armour regen cooldown.

        Parameters
        ----------
        value: float
            The new armour regen cooldown value.
        """
        self._entity_state["armour regen cooldown"] = value

    @property
    def bonus_attack_cooldown(self) -> float:
        """
        Gets the entity's bonus attack cooldown

        Returns
        -------
        float
            The entity's bonus attack cooldown.
        """
        return self._entity_state["bonus attack cooldown"]

    @bonus_attack_cooldown.setter
    def bonus_attack_cooldown(self, value: float) -> None:
        """
        Sets the entity's bonus attack cooldown.

        Parameters
        ----------
        value: float
            The new bonus attack cooldown.
        """
        self._entity_state["bonus attack cooldown"] = value

    def _initialise_entity_state(self) -> dict[str, float]:
        """
        Initialises the entity's state dict.

        Returns
        -------
        dict[str, float]
            The initialised entity state.
        """
        raise NotImplementedError

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
        self.update_indicator_bars()
        logger.debug(f"Dealing {damage} to {self}")

        # Check if the entity should be killed
        if self.health <= 0:
            self.remove_from_sprite_lists()

            # Remove the health and armour bar
            self.health_bar.background_box.remove_from_sprite_lists()
            self.health_bar.full_box.remove_from_sprite_lists()
            self.armour_bar.background_box.remove_from_sprite_lists()
            self.armour_bar.full_box.remove_from_sprite_lists()
            logger.info(f"Killed {self}")

    def regenerate_armour(self, delta_time: float) -> None:
        """
        Regenerates the entity's armour if they are able to do so.

        Parameters
        ----------
        delta_time:
            Time interval since the last time the function was called.
        """
        # Check if the entity has been out of combat for ARMOUR_REGEN_WAIT seconds
        if self.time_out_of_combat >= ARMOUR_REGEN_WAIT:
            # Check if enough has passed since the last armour regen
            if self.time_since_armour_regen >= self.armour_regen_cooldown:
                # Check if the entity's armour is below the max value
                if self.armour < self.max_armour:
                    # Regen armour
                    self.armour += ARMOUR_REGEN_AMOUNT
                    self.time_since_armour_regen = 0
                    self.update_indicator_bars()
                    logger.debug(f"Regenerated armour for {self}")
            else:
                # Increment the counter since not enough time has passed
                self.time_since_armour_regen += delta_time
        else:
            # Increment the counter since not enough time has passed
            self.time_out_of_combat += delta_time

    def update_indicator_bars(self) -> None:
        """Updates the entity's indicator bars."""
        try:
            self.health_bar.fullness = self.health / self.max_health
            self.armour_bar.fullness = self.armour / self.max_armour
        except ValueError:
            # Entity is already dead
            pass

    def attack(self) -> None:
        """
        Runs the entity's current attack algorithm.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        """
        raise NotImplementedError


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
    item_text: str = "Press R to activate"

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
        """
        Gets the player object for ease of access.

        Returns
        -------
        Player
            The player object.
        """
        # Make sure the player object is valid
        assert self.game.player is not None

        # Return the player object
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
