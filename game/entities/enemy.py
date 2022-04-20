from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from game.constants.entity import (
    ARMOUR_BAR_OFFSET,
    ENEMY1,
    FACING_LEFT,
    FACING_RIGHT,
    HEALTH_BAR_OFFSET,
    SPRITE_SIZE,
    AttackAlgorithmType,
    EntityID,
)
from game.entities.base import Entity, IndicatorBar

if TYPE_CHECKING:
    from game.constants.entity import BaseData
    from game.entities.movement import AIMovementBase
    from game.views.game import Game

# Get the logger
logger = logging.getLogger(__name__)


class Enemy(Entity):
    """
    Represents a hostile character in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the enemy can have a reference to it.
    x: int
        The x position of the enemy in the game map.
    y: int
        The y position of the enemy in the game map.

    Attributes
    ----------
    ai: AIMovementBase
        The AI movement algorithm which this entity uses.
    health_bar: IndicatorBar
        An indicator bar object which displays the enemy's health visually.
    armour_bar: IndicatorBar
        An indicator bar object which displays the enemy's armour visually.
    line_of_sight: bool
        Whether the enemy has line of sight with the player or not
    """

    # Class variables
    entity_id: EntityID = EntityID.ENEMY
    enemy_level: int = 1

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)
        self.ai: AIMovementBase = self.enemy_data.movement_algorithm.value(self)
        self.health_bar: IndicatorBar = IndicatorBar(
            self, (self.center_x, self.center_y + HEALTH_BAR_OFFSET), arcade.color.RED
        )
        self.armour_bar: IndicatorBar = IndicatorBar(
            self,
            (self.center_x, self.center_y + ARMOUR_BAR_OFFSET),
            arcade.color.SILVER,
        )
        self.line_of_sight: bool = False

    def __repr__(self) -> str:
        return f"<Enemy (Position=({self.center_x}, {self.center_y}))>"

    def on_update(self, delta_time: float = 1 / 60) -> None:
        """
        Processes enemy logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Make sure variables needed are valid
        assert self.game.player is not None

        # Update the enemy's time since last attack
        self.time_since_last_attack += delta_time

        # Check if the enemy is not in combat
        if not self.line_of_sight:
            # Enemy not in combat so check if they can regenerate armour
            if self.entity_data.armour_regen:
                self.check_armour_regen(delta_time)

                # Make sure the enemy's armour does not go over the maximum
                if self.armour > self.max_armour:
                    self.armour = self.max_armour
        else:
            # Enemy in combat so reset their combat counter
            self.time_out_of_combat = 0
            self.time_since_armour_regen = self.armour_regen_cooldown

        # Process the enemy's movement and get the force needed to move the enemy
        horizontal, vertical = self.ai.calculate_movement(self.game.player)

        # Set the needed internal variables
        self.facing = FACING_LEFT if horizontal < 0 else FACING_RIGHT
        self.direction = math.degrees(math.atan2(vertical, horizontal))

        # Apply the force
        self.physics_engines[0].apply_force(self, (horizontal, vertical))
        logger.debug(f"Applied force ({horizontal}, {vertical}) to {self}")

        # Update the health and armour bar's position
        self.health_bar.position = self.center_x, self.center_y + HEALTH_BAR_OFFSET
        self.armour_bar.position = self.center_x, self.center_y + ARMOUR_BAR_OFFSET

        # Make the enemy attack (they may not if the player is not within range)
        self.attack()

    def check_line_of_sight(self) -> bool:
        """
        Checks if the enemy has line of sight with the player.

        Returns
        -------
        bool
            Whether the enemy has line of sight with the player or not.
        """
        # Make sure variables needed are valid
        assert self.game.player is not None

        # Check for line of sight
        self.line_of_sight = arcade.has_line_of_sight(
            (self.center_x, self.center_y),
            (self.game.player.center_x, self.game.player.center_y),
            self.game.wall_sprites,
            self.enemy_data.view_distance * SPRITE_SIZE,
        )
        if self.line_of_sight:
            self.time_out_of_combat = 0

        # Return the result
        return self.line_of_sight

    def check_attack(self) -> bool:
        """
        Checks if the player is within a certain distance of the enemy and that the
        enemy can attack.

        Returns
        -------
        bool
            Whether the player is within distance of the enemy or not.
        """
        # Make sure variables needed are valid
        assert self.game.player is not None

        x_diff_squared = (self.game.player.center_x - self.center_x) ** 2
        y_diff_squared = (self.game.player.center_y - self.center_y) ** 2
        hypot_distance = math.sqrt(x_diff_squared + y_diff_squared)
        logger.info(f"{self} has distance of {hypot_distance} to {self.game.player}")
        return (
            hypot_distance <= self.enemy_data.attack_range * SPRITE_SIZE
            and self.time_since_last_attack
            >= (self.current_attack.attack_cooldown + self.bonus_attack_cooldown)
        )

    def update_indicator_bars(self) -> None:
        """Performs actions that should happen after the enemy takes damage."""
        # Update the health and armour bar
        try:
            self.health_bar.fullness = self.health / self.max_health
            self.armour_bar.fullness = self.armour / self.max_armour
        except ValueError:
            # Enemy is already dead
            pass

    def remove_indicator_bars(self) -> None:
        """Removes the indicator bars after the entity is killed."""
        # Remove the health and armour bar
        self.health_bar.background_box.remove_from_sprite_lists()
        self.health_bar.full_box.remove_from_sprite_lists()
        self.armour_bar.background_box.remove_from_sprite_lists()
        self.armour_bar.full_box.remove_from_sprite_lists()

    def attack(self) -> None:
        """Runs the enemy's current attack algorithm."""
        # Check if the player is within range of the enemy
        if not self.check_attack():
            return

        # Enemy can attack so reset the counters and determine what attack algorithm is
        # selected
        self.time_since_last_attack = 0
        match type(self.current_attack):
            case AttackAlgorithmType.RANGED.value:
                self.current_attack.process_attack(self.game.bullet_sprites)
            case AttackAlgorithmType.MELEE.value:
                print("enemy melee")
            case AttackAlgorithmType.AREA_OF_EFFECT.value:
                self.current_attack.process_attack(self.game.player)


class Enemy1(Enemy):
    """
    Represents the first enemy type in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the enemy can have a reference to it.
    x: int
        The x position of the enemy in the game map.
    y: int
        The y position of the enemy in the game map.
    """

    # Class variables
    entity_type: BaseData = ENEMY1

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
    ) -> None:
        super().__init__(game, x, y)
