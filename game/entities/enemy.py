"""Stores the enemy object which is hostile to the player."""
from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from game.constants.game_object import (
    ARMOUR_INDICATOR_BAR_COLOR,
    ENEMY_INDICATOR_BAR_OFFSET,
    FACING_LEFT,
    FACING_RIGHT,
    HEALTH_INDICATOR_BAR_COLOR,
    SPRITE_SIZE,
    AttackAlgorithmType,
    EntityAttributeType,
    ObjectID,
)
from game.entities.attribute import EntityAttribute
from game.entities.base import Entity, IndicatorBar
from game.entities.movement import EnemyMovementManager

if TYPE_CHECKING:
    from game.constants.game_object import BaseData, EnemyData
    from game.views.game_view import Game

__all__ = ("Enemy",)

# Get the logger
logger = logging.getLogger(__name__)


class Enemy(Entity):
    """Represents a hostile character in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the enemy can have a reference to it.
    x: int
        The x position of the enemy in the game map.
    y: int
        The y position of the enemy in the game map.
    enemy_level: int
        The level of this enemy.
    enemy_type: BaseData
        The raw data for this enemy.

    Attributes
    ----------
    movement_ai: EnemyMovementManager
        The movement AI class used for processing the logic needed for the enemy to
        move.
    player_within_range: bool
        Whether the player is within view distance of the enemy or not.
    """

    # Class variables
    object_id: ObjectID = ObjectID.ENEMY

    def __init__(
        self, game: Game, x: int, y: int, enemy_type: BaseData, enemy_level: int
    ) -> None:
        self.enemy_level: int = enemy_level
        super().__init__(game, x, y, enemy_type)
        self.movement_ai: EnemyMovementManager = EnemyMovementManager(self)
        self.health_bar: IndicatorBar = IndicatorBar(
            self,
            self.game.enemy_indicator_bar_sprites,
            (0, 0),
            HEALTH_INDICATOR_BAR_COLOR,
        )
        self.armour_bar: IndicatorBar = IndicatorBar(
            self,
            self.game.enemy_indicator_bar_sprites,
            (0, 0),
            ARMOUR_INDICATOR_BAR_COLOR,
        )
        self.player_within_range: bool = False

    def __repr__(self) -> str:
        return (
            f"<Enemy (Position=({self.center_x}, {self.center_y})) (Enemy"
            f" level={self.enemy_level})>"
        )

    @property
    def enemy_data(self) -> EnemyData:
        """Gets the enemy data if it exists.

        Returns
        -------
        EnemyData
            The enemy data.
        """
        # Make sure the enemy data is valid
        assert self.entity_type.enemy_data is not None

        # Return the enemy data
        return self.entity_type.enemy_data

    def _initialise_entity_state(self) -> dict[EntityAttributeType, EntityAttribute]:
        """Initialises the entity's state dict.

        Returns
        -------
        dict[EntityAttributeType, EntityAttribute]
            The initialised entity state.
        """
        # Get the enemy level adjusted for an array index
        adjusted_level = self.enemy_level - 1
        logger.debug("Initialising enemy with level %d", adjusted_level)

        # Create the entity state dict
        return {
            attribute_type: EntityAttribute(self, attribute_data, adjusted_level)
            for attribute_type, attribute_data in self.attribute_data.items()
        }
        # return {
        #     "health": self.attribute_data[EntityAttributeType.HEALTH].increase(
        #         adjusted_level
        #     ),
        #     "max health": self.attribute_data[EntityAttributeType.HEALTH].increase(
        #         adjusted_level
        #     ),
        #     "armour": self.attribute_data[EntityAttributeType.ARMOUR].increase(
        #         adjusted_level
        #     ),
        #     "max armour": self.attribute_data[EntityAttributeType.ARMOUR].increase(
        #         adjusted_level
        #     ),
        #     "max velocity": self.attribute_data[EntityAttributeType.SPEED].increase(
        #         adjusted_level
        #     ),
        #     "armour regen cooldown": self.attribute_data[
        #         EntityAttributeType.REGEN_COOLDOWN
        #     ].increase(adjusted_level),
        #     "bonus attack cooldown": 0,
        # }

    def post_on_update(self, delta_time: float) -> None:
        """Processes enemy logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Make sure variables needed are valid
        assert self.game.player is not None
        assert self.armour_bar is not None
        assert self.health_bar is not None

        # Check if the enemy is not in combat
        if not self.player_within_range:
            # Enemy not in combat so check if they can regenerate armour
            if self.entity_data.armour_regen:
                self.regenerate_armour(delta_time)
        else:
            # Enemy in combat so reset their combat counter
            self.time_out_of_combat = 0
            self.time_since_armour_regen = self.armour_regen_cooldown.value

        # Make the enemy move
        self.move(delta_time)

        # Make the enemy attack (they may not if the player is not within range)
        self.attack()

    def move(self, delta_time: float) -> None:
        """Processes the needed actions for the enemy to move.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Make sure variables needed are valid
        assert self.game.player is not None

        # Determine what movement algorithm to use based on the distance to the player
        player_tile_distance = (
            math.sqrt(
                (self.center_y - self.game.player.center_y) ** 2
                + (self.center_x - self.game.player.center_x) ** 2
            )
        ) / SPRITE_SIZE
        if player_tile_distance > self.enemy_data.view_distance:
            # Player is outside the enemy's view distance so have them wander around
            self.player_within_range = False
            horizontal, vertical = self.movement_ai.calculate_wander_force()
        else:
            # Player is within the enemy's view distance use the vector field to move
            # towards the player
            self.player_within_range = True
            self.time_out_of_combat = 0
            horizontal, vertical = self.movement_ai.calculate_vector_field_force()

        # Set the needed internal variables
        self.facing = FACING_LEFT if horizontal < 0 else FACING_RIGHT
        angle = math.degrees(math.atan2(vertical, horizontal))
        if angle < 0:
            angle += 360
        self.direction = angle

        # Apply the force
        self.physics_engines[0].apply_force(self, (horizontal, vertical))

        # Update the health and armour bar's position
        self.armour_bar.position = (
            self.center_x,
            self.center_y + ENEMY_INDICATOR_BAR_OFFSET,
        )
        self.health_bar.position = (
            self.center_x,
            self.armour_bar.top + (self.health_bar.bar_height / 2),
        )

    def attack(self) -> None:
        """Runs the enemy's current attack algorithm."""
        # Make sure variables needed are valid
        assert self.game.player is not None

        # Check if the player is within range and line of sight of the enemy
        if not (
            self.check_line_of_sight(self.current_attack.attack_range)
            and self.player_within_range
            and self.time_since_last_attack
            >= (self.current_attack.attack_cooldown + self.bonus_attack_cooldown.value)
        ):
            return

        # Enemy can attack so reset the counters and determine what attack algorithm is
        # selected
        self.time_since_last_attack = 0
        match type(self.current_attack):
            case AttackAlgorithmType.RANGED.value:
                self.current_attack.process_attack(self.game.bullet_sprites)
            case AttackAlgorithmType.MELEE.value:
                self.current_attack.process_attack([self.game.player])
            case AttackAlgorithmType.AREA_OF_EFFECT.value:
                self.current_attack.process_attack(self.game.player)

    def check_line_of_sight(self, max_tile_range: int) -> bool:
        """Checks if the enemy has line of sight with the player.

        Parameters
        ----------
        max_tile_range: int
            The max tile distance that the player can be from the enemy.

        Returns
        -------
        bool
            Whether the enemy has line of sight with the player or not.
        """
        # Make sure variables needed are valid
        assert self.game.player is not None

        # Check for line of sight
        return arcade.has_line_of_sight(
            (self.center_x, self.center_y),
            (self.game.player.center_x, self.game.player.center_y),
            self.game.wall_sprites,
            max_tile_range * SPRITE_SIZE,
        )
