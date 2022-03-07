from __future__ import annotations

# Pip
import arcade

# Custom
from entities.entity import Entity


class Player(Entity):
    """
    Represents the player character in the game.

    Parameters
    ----------
    x: int
        The x position of the player in the game map.
    y: int
        The y position of the player in the game map.
    texture_dict: dict[str, list[list[arcade.Texture]]]
        The textures which represent this player.
    health: int
        The health of this player.

    Attributes
    ----------
    """

    def __init__(
        self,
        x: int,
        y: int,
        texture_dict: dict[str, list[list[arcade.Texture]]],
        health: int,
    ) -> None:
        super().__init__(x, y, texture_dict, health)
