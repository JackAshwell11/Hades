from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Optional

# Pip
import arcade

# Custom
from entities.ai import FollowLineOfSight

if TYPE_CHECKING:
    from .entity import Entity


class Character:
    """
    Represents an enemy or playable character in the game.

    Parameters
    ----------
    ai: Optional[FollowLineOfSight]
        The AI algorithm which this character uses.
    """

    def __init__(self, ai: Optional[FollowLineOfSight] = None) -> None:
        self.owner: Optional[Entity] = None
        self.ai: Optional[FollowLineOfSight] = ai
        self.time_since_last_attack: float = 0

    def __repr__(self) -> str:
        return "<Character>"

    def ranged_attack(self, x: float, y: float, bullet_list: arcade.SpriteList) -> None:
        """Performs an attack in the direction the character is facing."""
        # Make sure variables needed are valid
        assert self.owner is not None

        # Reset the internal time counter
        self.time_since_last_attack = 0

        # Create the bullet sprite
        # bullet = arcade.Sprite(texture=textures["attack"][0],
        # center_x=self.owner.center_x, center_y=self.owner.center_y)

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
