from __future__ import annotations

# Pip
import arcade

# Custom
from entities.ai import FollowLineOfSight
from entities.entity import Entity


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
