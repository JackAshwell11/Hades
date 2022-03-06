from __future__ import annotations

# Builtin
import math
from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Tuple

# Pip
import arcade
from constants import BULLET_OFFSET, BULLET_VELOCITY

# Custom
from entities.ai import FollowLineOfSight

if TYPE_CHECKING:
    from entities.entity import Entity
    from physics import PhysicsEngine


class Bullet(arcade.SpriteSolidColor):
    """
    Represents a bullet in the game.

    Parameters
    ----------
    x: float
        The starting x position of the bullet.
    y: float
        The starting y position of the bullet.
    width: int
        Width of the bullet.
    height: int
        Height of the bullet.
    color: Tuple[int, int, int]
        The color of the bullet.
    """

    def __init__(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        color: Tuple[int, int, int],
    ) -> None:
        super().__init__(width=width, height=height, color=color)
        self.center_x: float = x
        self.center_y: float = y

    def __repr__(self) -> str:
        return f"<Bullet (Position=({self.center_x}, {self.center_y}))>"


class Damage(IntEnum):
    """Stores the amount of damage each character deals to other characters."""

    PLAYER = 20
    ENEMY = 10


class Character:
    """
    Represents an enemy or playable character in the game.

    Parameters
    ----------
    health: int
        The health of this character.
    ai: Optional[FollowLineOfSight]
        The AI algorithm which this character uses.

    Attributes
    ----------
    owner: Optional[Entity]
        The parent entity object which manages this character.
    time_since_last_attack: float
        How long it has been since the last attack.
    """

    def __init__(self, health: int, ai: Optional[FollowLineOfSight] = None) -> None:
        self.health: int = health
        self.ai: Optional[FollowLineOfSight] = ai
        self.owner: Optional[Entity] = None
        self.time_since_last_attack: float = 0

    def __repr__(self) -> str:
        return "<Character>"

    def deal_damage(self, damage: Damage) -> None:
        """
        Deals damage to a character.

        Parameters
        ----------
        damage: Damage
            The damage to deal to the entity.
        """
        self.health -= damage.value

    def melee_attack(self, enemy_list: arcade.SpriteList, damage: Damage) -> None:
        """
        Performs a melee attack dealing damage to every character that is within the
        attack range (the cone).

        Parameters
        ----------
        enemy_list: arcade.SpriteList
            Who the character should attack.
        damage: Damage
            The amount of damage to deal.
        """
        for enemy in self.owner.cone.collides_with_list(enemy_list):  # type: ignore
            enemy.character.deal_damage(damage)

    def ranged_attack(self, bullet_list: arcade.SpriteList) -> None:
        """
        Performs a ranged attack in the direction the character is facing.

        Parameters
        ----------
        bullet_list: arcade.SpriteList
            The sprite list to add the bullet to.
        """
        # Make sure variables needed are valid
        assert self.owner is not None

        # Reset the time counter
        self.time_since_last_attack = 0

        # Create and add the new bullet to the physics engine
        new_bullet = Bullet(
            self.owner.center_x, self.owner.center_y, 25, 5, arcade.color.RED
        )
        new_bullet.angle = self.owner.direction
        physics: PhysicsEngine = self.owner.physics_engines[0]
        physics.add_bullet(new_bullet)
        bullet_list.append(new_bullet)

        # Move the bullet away from the entity a bit to stop its colliding with them
        angle_radians = self.owner.direction * math.pi / 180
        new_x, new_y = (
            new_bullet.center_x + math.cos(angle_radians) * BULLET_OFFSET,
            new_bullet.center_y + math.sin(angle_radians) * BULLET_OFFSET,
        )
        physics.set_position(new_bullet, (new_x, new_y))

        # Calculate its velocity
        change_x, change_y = (
            math.cos(angle_radians) * BULLET_VELOCITY,
            math.sin(angle_radians) * BULLET_VELOCITY,
        )
        physics.set_velocity(new_bullet, (change_x, change_y))

    # def melee_attack(self) -> None:
    #     # Make sure variables needed are valid
    #     assert self.owner is not None
    #
    #     # Reset the internal time counter
    #     self.time_since_last_attack = 0
    #
    #     # Get the points for a triangle from the center of the character in the
    #     # direction the character is facing with a height of ATTACK_DISTANCE
    #     points = []
    #     if self.direction is Direction.NORTH:
    #         points = [
    #             (self.owner.center_x, self.owner.center_y),
    #             (
    #                 self.owner.center_x - ATTACK_DISTANCE,
    #                 self.owner.center_y + ATTACK_DISTANCE,
    #             ),
    #             (
    #                 self.owner.center_x + ATTACK_DISTANCE,
    #                 self.owner.center_y + ATTACK_DISTANCE,
    #             ),
    #         ]
    #     elif self.direction is Direction.SOUTH:
    #         points = [
    #             (self.owner.center_x, self.owner.center_y),
    #             (
    #                 self.owner.center_x - ATTACK_DISTANCE,
    #                 self.owner.center_y - ATTACK_DISTANCE,
    #             ),
    #             (
    #                 self.owner.center_x + ATTACK_DISTANCE,
    #                 self.owner.center_y - ATTACK_DISTANCE,
    #             ),
    #         ]
    #     elif self.direction is Direction.EAST:
    #         points = [
    #             (self.owner.center_x, self.owner.center_y),
    #             (
    #                 self.owner.center_x + ATTACK_DISTANCE,
    #                 self.owner.center_y + ATTACK_DISTANCE,
    #             ),
    #             (
    #                 self.owner.center_x + ATTACK_DISTANCE,
    #                 self.owner.center_y - ATTACK_DISTANCE,
    #             ),
    #         ]
    #     elif self.direction is Direction.WEST:
    #         points = [
    #             (self.owner.center_x, self.owner.center_y),
    #             (
    #                 self.owner.center_x - ATTACK_DISTANCE,
    #                 self.owner.center_y + ATTACK_DISTANCE,
    #             ),
    #             (
    #                 self.owner.center_x - ATTACK_DISTANCE,
    #                 self.owner.center_y - ATTACK_DISTANCE,
    #             ),
    #         ]
    #
    #     # Create the triangle shape
    #     arcade.Sprite()
    #
    #     print(points)
    #
    #     # triangle = arcade.create_polygon(points, arcade.color.BLUE)
