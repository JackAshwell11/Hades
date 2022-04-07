from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants.entity import FACING_LEFT, FACING_RIGHT
from constants.enums import AttackAlgorithmType, EntityID
from constants.general import SPRITE_SIZE
from entities.base import Entity

if TYPE_CHECKING:
    from constants.entity import BaseType
    from entities.movement import AIMovementBase
    from views.game import Game

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
    enemy_type: BaseType
        The constant data about this specific enemy.

    Attributes
    ----------
    ai: AIMovementBase
        The AI movement algorithm which this entity uses.
    line_of_sight: bool
        Whether the enemy has line of sight with the player or not
    """

    # Class variables
    ID: EntityID = EntityID.ENEMY

    def __init__(
        self,
        game: Game,
        x: int,
        y: int,
        enemy_type: BaseType,
    ) -> None:
        super().__init__(game, x, y, enemy_type)
        self.ai: AIMovementBase = self.custom_data.movement_algorithm.value(self)
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
        # Update the enemy's time since last attack
        self.time_since_last_attack += delta_time

        # Check if the enemy is not in combat
        if not self.line_of_sight:
            # Enemy not in combat so check if they can regenerate armour
            if self.entity_type.armour_regen:
                self.check_armour_regen(delta_time)

                # Make sure the enemy's armour does not go over the maximum
                self.armour: int  # Mypy gives self.armour an undetermined type error
                if self.armour > self.entity_type.armour:
                    self.armour = self.entity_type.armour
            return

        # Enemy in combat so reset their combat counter
        self.time_out_of_combat = 0
        self.time_since_armour_regen = self.entity_type.armour_regen_cooldown

        # Player is within line of sight so get the force needed to move the enemy
        horizontal, vertical = self.ai.calculate_movement(
            self.game.player, self.game.wall_sprites
        )

        # Set the needed internal variables
        self.facing = FACING_LEFT if horizontal < 0 else FACING_RIGHT
        self.direction = math.degrees(math.atan2(vertical, horizontal))

        # Apply the force
        self.physics_engines[0].apply_force(self, (horizontal, vertical))
        logger.debug(f"Applied force ({horizontal}, {vertical}) to {self}")

        # Check if the player is within the attack range and can attack
        x_diff_squared = (self.game.player.center_x - self.center_x) ** 2
        y_diff_squared = (self.game.player.center_y - self.center_y) ** 2
        hypot_distance = math.sqrt(x_diff_squared + y_diff_squared)
        if (
            hypot_distance <= self.custom_data.attack_range * SPRITE_SIZE
            and self.time_since_last_attack >= self.entity_type.attack_cooldown
        ):
            # Enemy can attack so reset the counter and attack
            logger.info(f"{self} attacking player with distance {hypot_distance}")
            self.time_since_last_attack: float = 0  # Mypy is annoying
            self.attack()

    def check_line_of_sight(self) -> bool:
        """
        Checks if the enemy has line of sight with the player.

        Returns
        -------
        bool
            Whether the enemy has line of sight with the player or not.
        """
        # Check for line of sight
        self.line_of_sight = arcade.has_line_of_sight(
            (self.center_x, self.center_y),
            (self.game.player.center_x, self.game.player.center_y),
            self.game.wall_sprites,
            self.custom_data.view_distance * SPRITE_SIZE,
        )
        if self.line_of_sight:
            self.time_out_of_combat = 0

        # Return the result
        return self.line_of_sight

    def attack(self) -> None:
        """Runs the enemy's current attack algorithm."""
        # Find out what attack algorithm is selected
        match type(self.current_attack):
            case AttackAlgorithmType.RANGED.value:
                self.current_attack.process_attack(self.game.bullet_sprites)
            case AttackAlgorithmType.MELEE.value:
                print("enemy melee")
            case AttackAlgorithmType.AREA_OF_EFFECT.value:
                print("enemy aoe")
