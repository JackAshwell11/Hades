from __future__ import annotations

# Builtin
import math
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants import (
    ATTACK_COOLDOWN,
    ENEMY_ATTACK_RANGE,
    ENEMY_VIEW_DISTANCE,
    SPRITE_WIDTH,
)
from entities.ai import FollowLineOfSight
from entities.entity import Entity, EntityID

if TYPE_CHECKING:
    from physics import PhysicsEngine
    from views.game import Game


class Enemy(Entity):
    """
    Represents a hostile character in the game.

    Parameters
    ----------
    x: int
        The x position of the enemy in the game map.
    y: int
        The y position of the enemy in the game map.
    texture_dict: dict[str, list[list[arcade.Texture]]]
        The textures which represent this enemy.
    health: int
        The health of this enemy.
    ai: FollowLineOfSight
        The AI which this entity uses.
    """

    # Class variables
    ID: EntityID = EntityID.ENEMY

    def __init__(
        self,
        x: int,
        y: int,
        texture_dict: dict[str, list[list[arcade.Texture]]],
        health: int,
        ai: FollowLineOfSight,
    ) -> None:
        super().__init__(x, y, texture_dict, health)
        self.ai: FollowLineOfSight = ai
        self.ai.owner = self

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
        # Check if the enemy should be killed
        if self.health <= 0:
            self.remove_from_sprite_lists()
            return

        # Update the enemy's time since last attack
        self.time_since_last_attack += delta_time

        # Get the current view and the physics engine.
        # The current view will be used to get the player object and the wall sprites
        current_view: Game = arcade.get_window().current_view  # noqa
        physics: PhysicsEngine = self.physics_engines[0]

        # Check if the player is within the max view distance
        if arcade.has_line_of_sight(
            (self.center_x, self.center_y),
            (current_view.player.center_x, current_view.player.center_y),
            current_view.wall_sprites,
            ENEMY_VIEW_DISTANCE * SPRITE_WIDTH,
        ):
            # Get the force needed to move the enemy
            horizontal, vertical = self.ai.calculate_movement(
                current_view.player, current_view.wall_sprites
            )

            # Set the needed internal variables
            self.facing = 1 if horizontal < 0 else 0
            self.direction = math.degrees(math.atan2(vertical, horizontal))

            # Apply the force
            physics.apply_force(self, (horizontal, vertical))

            # Check if the player is within the attack range and can attack
            x_diff_squared = (current_view.player.center_x - self.center_x) ** 2
            y_diff_squared = (current_view.player.center_y - self.center_y) ** 2
            hypot_distance = math.sqrt(x_diff_squared + y_diff_squared)
            if (
                hypot_distance <= ENEMY_ATTACK_RANGE * SPRITE_WIDTH
                and self.time_since_last_attack >= ATTACK_COOLDOWN
            ):
                # Enemy can attack so reset the counter and attack
                self.time_since_last_attack: float = 0  # Mypy is annoying
                self.ranged_attack(current_view.bullet_sprites)
