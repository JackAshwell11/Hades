from __future__ import annotations

# Pip
import arcade

# Custom
from entities.entity import Entity, EntityID


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
    """

    # Class variables
    ID: EntityID = EntityID.PLAYER

    def __init__(
        self,
        x: int,
        y: int,
        texture_dict: dict[str, list[list[arcade.Texture]]],
        health: int,
    ) -> None:
        super().__init__(x, y, texture_dict, health)

    def __repr__(self) -> str:
        return f"<Player (Position=({self.center_x}, {self.center_y}))>"

    def on_update(self, delta_time: float = 1 / 60) -> None:
        """
        Processes player logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the player's time since last attack
        self.time_since_last_attack += delta_time
