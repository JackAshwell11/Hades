"""Manages the base classes used by all game objects."""
from __future__ import annotations

# Builtin
import contextlib
import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from hades.constants.game_objects import (
    ARMOUR_REGEN_AMOUNT,
    ARMOUR_REGEN_WAIT,
    SPRITE_SCALE,
    EntityAttributeType,
    ObjectID,
)
from hades.game_objects.attacks import create_attack
from hades.textures import grid_pos_to_pixel

if TYPE_CHECKING:
    from hades.constants.game_objects import (
        AttackAlgorithmType,
        AttackData,
        BaseData,
        EntityAttributeData,
        EntityData,
    )
    from hades.game_objects.attacks import AttackBase
    from hades.game_objects.attributes import EntityAttribute
    from hades.game_objects.players import Player
    from hades.physics import PhysicsEngine
    from hades.views.game_view import Game

__all__ = (
    "CollectibleTile",
    "Entity",
    "IndicatorBar",
    "InteractiveTile",
    "Tile",
    "UsableTile",
)

# Get the logger
logger = logging.getLogger(__name__)


class IndicatorBar:
    """Represents a bar which can display information about an entity.

    Parameters
    ----------
    owner: Entity
        The owner of this indicator bar.
    target_spritelist: arcade.SpriteList
        The spritelist that the indicator bar sprites should be added too.
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
    scale: float
        The scale of the indicator bar.
    """

    __slots__ = (
        "owner",
        "target_spritelist",
        "_bar_width",
        "_bar_height",
        "_center_x",
        "_center_y",
        "_fullness",
        "_scale",
        "_background_box",
        "_full_box",
    )

    def __init__(
        self,
        owner: Entity,
        target_spritelist: arcade.SpriteList,
        position: tuple[float, float] = (0, 0),
        full_color: arcade.Color = arcade.color.GREEN,
        background_color: arcade.Color = arcade.color.BLACK,
        width: int = 50,
        height: int = 4,
        border_size: int = 4,
        scale: float = 1.0,
    ) -> None:
        # Store the reference to the owner and the target spritelist
        self.owner: Entity = owner
        self.target_spritelist: arcade.SpriteList = target_spritelist

        # Set the needed size variables
        self._bar_width: int = width
        self._bar_height: int = height
        self._center_x: float = 0.0
        self._center_y: float = 0.0
        self._fullness: float = 0.0
        self._scale: float = 1.0

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
        self.target_spritelist.append(self.background_box)
        self.target_spritelist.append(self.full_box)

        # Set the fullness, position and scale of the bar
        self.fullness = 1.0
        self.position = position
        self.scale = scale

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<IndicatorBar (Owner={self.owner})>"

    @property
    def background_box(self) -> arcade.SpriteSolidColor:
        """Get the background box object of the indicator bar.

        Returns
        -------
        arcade.SpriteSolidColor
            The background box object.
        """
        return self._background_box

    @property
    def full_box(self) -> arcade.SpriteSolidColor:
        """Get the full box of the indicator bar.

        Returns
        -------
        arcade.SpriteSolidColor
            The full box object.
        """
        return self._full_box

    @property
    def bar_width(self) -> int:
        """Get the width of the bar.

        Returns
        -------
        int
            The width of the bar.
        """
        return self._bar_width

    @property
    def bar_height(self) -> int:
        """Get the height of the bar.

        Returns
        -------
        int
            The height of the bar.
        """
        return self._bar_height

    @property
    def center_x(self) -> float:
        """Get the x position of the bar.

        Returns
        -------
        float
            The x position of the bar.
        """
        return self._center_x

    @property
    def center_y(self) -> float:
        """Get the y position of the bar.

        Returns
        -------
        float
            The y position of the bar.
        """
        return self._center_y

    @property
    def top(self) -> float:
        """Get the y coordinate of the top of the bar.

        Returns
        -------
        float
            The y coordinate of the top of the bar.
        """
        return self.background_box.top

    @property
    def bottom(self) -> float:
        """Get the y coordinate of the bottom of the bar.

        Returns
        -------
        float
            The y coordinate of the bottom of the bar.
        """
        return self.background_box.bottom

    @property
    def left(self) -> float:
        """Get the x coordinate of the left of the bar.

        Returns
        -------
        float
            The x coordinate of the left of the bar.
        """
        return self.background_box.left

    @property
    def right(self) -> float:
        """Get the x coordinate of the right of the bar.

        Returns
        -------
        float
            The x coordinate of the right of the bar.
        """
        return self.background_box.right

    @property
    def fullness(self) -> float:
        """Get the fullness of the bar.

        Returns
        -------
        float
            The fullness of the bar.
        """
        return self._fullness

    @fullness.setter
    def fullness(self, new_fullness: float) -> None:
        """Set the fullness of the bar.

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
        if new_fullness < 0.0 or new_fullness > 1.0:
            raise ValueError(
                f"Got {new_fullness}, but fullness must be between 0.0 and 1.0."
            )

        # Set the size of the bar
        self._fullness = new_fullness
        if new_fullness == 0.0:
            # Set the full_box to not be visible since it is not full anymore
            self.full_box.visible = False
        else:
            # Set the full_box to be visible in case it wasn't then update the bar
            self.full_box.visible = True
            self.full_box.width = self.bar_width * new_fullness * self.scale
            self.full_box.left = self.center_x - (self.bar_width / 2) * self.scale

    @property
    def position(self) -> tuple[float, float]:
        """Get the current position of the bar.

        Returns
        -------
        tuple[float, float]
            The current position of the bar.
        """
        return self.center_x, self.center_y

    @position.setter
    def position(self, new_position: tuple[float, float]) -> None:
        """Set the new position of the bar.

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
            self.full_box.left = self.center_x - (self.bar_width / 2) * self.scale

    @property
    def scale(self) -> float:
        """Get the scale of the bar.

        Returns
        -------
        float
            The scale of the bar.
        """
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        """Set the new scale of the bar.

        Parameters
        ----------
        value: float
            The new scale of the bar.
        """
        # Check if the scale has changed. If so, change the bar's scale
        if value != self.scale:
            self._scale = value
            self.background_box.scale = value
            self.full_box.scale = value


class GameObject(arcade.Sprite):
    """The base class for all game objects.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the object can have a reference to it.
    x: int
        The x position of the object in the game map.
    y: int
        The y position of the object in the game map.

    Attributes
    ----------
    center_x: float
        The x position of the object on the screen.
    center_y: float
        The y position of the object on the screen.
    """

    # Class variables
    object_id: ObjectID = ObjectID.BASE

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(scale=SPRITE_SCALE)
        self.game: Game = game
        self.center_x, self.center_y = grid_pos_to_pixel(x, y)

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<GameObject (Position=({self.center_x}, {self.center_y}))>"


class Entity(GameObject, metaclass=ABCMeta):
    """Represents an entity in the game.

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
    entity_state: dict[EntityAttributeType, EntityAttribute]
        The entity's state which manages all the entity's attributes.
    attack_algorithms: list[AttackBase]
        A list of the entity's attack algorithms.
    health_bar: IndicatorBar | None
        An indicator bar object which displays the entity's health visually.
    armour_bar: IndicatorBar | None
        An indicator bar object which displays the entity's armour visually.
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

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
        entity_type: BaseData,
    ) -> None:
        super().__init__(game, x, y)
        self.entity_type: BaseData = entity_type
        self.texture: arcade.Texture = self.entity_data.textures["idle"][0][0]
        self.entity_state: dict[
            EntityAttributeType, EntityAttribute
        ] = self._initialise_entity_state()
        self.attack_algorithms: list[AttackBase] = [
            create_attack(self, attack_type, attack_data)
            for attack_type, attack_data in self.attacks.items()
        ]
        self.health_bar: IndicatorBar | None = None
        self.armour_bar: IndicatorBar | None = None
        self.current_attack_index: int = 0
        self.direction: float = 0
        self.facing: int = 0
        self.time_since_last_attack: float = 0
        self.time_out_of_combat: float = 0
        self.time_since_armour_regen: float = self.armour_regen_cooldown.value

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<Entity (Position=({self.center_x}, {self.center_y}))>"

    @property
    def entity_data(self) -> EntityData:
        """Get the general entity data.

        Returns
        -------
        EntityData
            The general entity data.
        """
        return self.entity_type.entity_data

    @property
    def attacks(self) -> dict[AttackAlgorithmType, AttackData]:
        """Get the entity's attacks.

        Returns
        -------
        dict[AttackAlgorithmType, AttackData]
            The entity's attacks.
        """
        return self.entity_type.attacks

    @property
    def attribute_data(self) -> dict[EntityAttributeType, EntityAttributeData]:
        """Get the entity's attribute data.

        Returns
        -------
        dict[EntityAttributeType, EntityAttributeData]
            The entity's attribute data.
        """
        return self.entity_data.attribute_data

    @property
    def current_attack(self) -> AttackBase:
        """Get the currently selected attack algorithm.

        Returns
        -------
        AttackBase
            The currently selected attack algorithm.
        """
        return self.attack_algorithms[self.current_attack_index]

    @property
    def physics(self) -> PhysicsEngine:
        """Get the entity's physics engine.

        Returns
        -------
        PhysicsEngine
            The entity's physics engine
        """
        return self.physics_engines[0]

    @property
    def health(self) -> EntityAttribute:
        """Get the entity's health.

        Returns
        -------
        EntityAttribute
            The entity's health
        """
        return self.entity_state[EntityAttributeType.HEALTH]

    @property
    def armour(self) -> EntityAttribute:
        """Get the entity's armour.

        Returns
        -------
        EntityAttribute
            The entity's armour.
        """
        return self.entity_state[EntityAttributeType.ARMOUR]

    @property
    def max_velocity(self) -> EntityAttribute:
        """Get the entity's max velocity.

        Returns
        -------
        EntityAttribute
            The entity's max velocity.
        """
        return self.entity_state[EntityAttributeType.SPEED]

    @property
    def armour_regen_cooldown(self) -> EntityAttribute:
        """Get the entity's armour regen cooldown.

        Returns
        -------
        EntityAttribute
            The entity's armour regen cooldown.
        """
        return self.entity_state[EntityAttributeType.REGEN_COOLDOWN]

    @property
    def fire_rate_penalty(self) -> EntityAttribute:
        """Get the entity's fire rate penalty.

        Returns
        -------
        EntityAttribute
            The entity's fire rate penalty.
        """
        return self.entity_state[EntityAttributeType.FIRE_RATE_PENALTY]

    def _initialise_entity_state(self) -> dict[EntityAttributeType, EntityAttribute]:
        """Initialise the entity's state dict.

        Raises
        ------
        NotImplementedError
            The function is not implemented.

        Returns
        -------
        dict[EntityAttributeType, EntityAttribute]
            The initialised entity state.
        """
        raise NotImplementedError

    def on_update(self, delta_time: float = 1 / 60) -> None:
        """Process enemy logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Make sure variables needed are valid
        assert self.game.vector_field is not None

        # Update the player's time since last attack
        self.time_since_last_attack += delta_time

        # Update any status effects
        for attribute in self.entity_state.values():
            if attribute.applied_status_effect:
                logger.debug(
                    "Updating status effect %r for entity %r",
                    attribute.applied_status_effect,
                    self,
                )
                attribute.update_status_effect(delta_time)

        # Run the entity's post on_update
        self.post_on_update(delta_time)

    def deal_damage(self, damage: int) -> None:
        """Deal damage to an entity.

        Parameters
        ----------
        damage: int
            The amount of health to take away from the entity.
        """
        # Make sure variables needed are valid
        assert self.health_bar is not None
        assert self.armour_bar is not None

        # Check if the entity still has armour
        if self.armour.value > 0:
            # Damage the armour
            self.armour.value -= damage
            if self.armour.value < 0:
                # Damage exceeds armour so damage health
                self.health.value += self.armour.value
                self.armour.value = 0
        else:
            # Damage the health
            self.health.value -= damage
        self.update_indicator_bars()
        logger.debug("Dealing %d to %r", damage, self)

        # Check if the entity should be killed
        if self.health.value <= 0:
            # Kill the entity
            self.remove_from_sprite_lists()

            # Remove the health and armour bar
            self.health_bar.background_box.remove_from_sprite_lists()
            self.health_bar.full_box.remove_from_sprite_lists()
            self.armour_bar.background_box.remove_from_sprite_lists()
            self.armour_bar.full_box.remove_from_sprite_lists()
            logger.info("Killed %r", self)

    def regenerate_armour(self, delta_time: float) -> None:
        """Regenerate the entity's armour if they are able to do so.

        Parameters
        ----------
        delta_time:
            Time interval since the last time the function was called.
        """
        # Check if the entity has been out of combat for ARMOUR_REGEN_WAIT seconds
        if self.time_out_of_combat >= ARMOUR_REGEN_WAIT:
            # Check if enough has passed since the last armour regen
            if self.time_since_armour_regen >= self.armour_regen_cooldown.value:
                # Check if the entity's armour is below the max value
                if self.armour.value < self.armour.max_value:
                    # Regen armour
                    self.armour.value = self.armour.value + ARMOUR_REGEN_AMOUNT
                    self.time_since_armour_regen = 0
                    self.update_indicator_bars()
                    logger.debug(
                        "Regenerated %d armour for %r", ARMOUR_REGEN_AMOUNT, self
                    )
            else:
                # Increment the counter since not enough time has passed
                self.time_since_armour_regen += delta_time
        else:
            # Increment the counter since not enough time has passed
            self.time_out_of_combat += delta_time

    def update_indicator_bars(self) -> None:
        """Update the entity's indicator bars."""
        # Make sure variables needed are valid
        assert self.health_bar is not None
        assert self.armour_bar is not None

        # Update the indicator bars
        with contextlib.suppress(ValueError):
            # If this fails, the entity is already dead
            new_health_fullness = self.health.value / self.health.max_value
            self.health_bar.fullness = new_health_fullness
            new_armour_fullness = self.armour.value / self.armour.max_value
            self.armour_bar.fullness = new_armour_fullness

    @abstractmethod
    def post_on_update(self, delta_time: float) -> None:
        """Process custom entity logic.

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

    @abstractmethod
    def move(self, delta_time: float) -> None:
        """Process the needed actions for the entity to move.

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

    @abstractmethod
    def attack(self) -> None:
        """Run the entity's current attack algorithm.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        """
        raise NotImplementedError


class Tile(GameObject):
    """Represents a tile that does not move in the game.

    Parameters
    ----------
    x: int
        The x position of the tile in the game map.
    y: int
        The y position of the tile in the game map.

    Attributes
    ----------
    texture: arcade.Texture
        The sprite which represents this tile.
    """

    # Class variables
    object_id: ObjectID = ObjectID.TILE
    raw_texture: arcade.Texture | None = None
    blocking: bool = False

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)
        self.texture: arcade.Texture = self.raw_texture

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<Tile (Position=({self.center_x}, {self.center_y}))>"


class InteractiveTile(Tile):
    """Represents a tile that can be interacted with by the player.

    This is only meant to be inherited from and should not be initialised on its own.
    """

    # Class variables
    item_text: str = ""

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<InteractiveTile (Position=({self.center_x}, {self.center_y}))>"

    @property
    def player(self) -> Player:
        """Get the player object for ease of access.

        Returns
        -------
        Player
            The player object.
        """
        # Make sure the player object is valid
        assert self.game.player is not None

        # Return the player object
        return self.game.player


class UsableTile(InteractiveTile):
    """Represents a tile that can be used/activated by the player."""

    # Class variables
    item_text: str = "Press R to activate"

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<UsableTile (Position=({self.center_x}, {self.center_y}))>"

    @abstractmethod
    def item_use(self) -> bool:
        """Process item use functionality.

        Returns
        -------
        bool
            Whether the item use was successful or not.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        """
        raise NotImplementedError


class CollectibleTile(InteractiveTile):
    """Represents a tile that can be picked up by the player."""

    # Class variables
    item_text: str = "Press E to pick up"

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<CollectibleTile (Position=({self.center_x}, {self.center_y}))>"

    def item_pick_up(self) -> bool:
        """Process item pick up functionality.

        Returns
        -------
        bool
            Whether the collectible pickup was successful or not.
        """
        # Try and add the item to the player's inventory
        if self.player.add_item_to_inventory(self):
            # Add successful
            self.remove_from_sprite_lists()

            # Activate was successful
            return True
        else:
            # Add not successful due to full inventory
            self.game.display_info_box("Inventory is full")
            return False
