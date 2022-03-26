from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from constants import PlayerType
from entities.base import Entity, EntityID
from entities.inventory import Inventory
from shaders.melee import MeleeShader

if TYPE_CHECKING:
    from entities.base import Item
    from views.game import Game


class Player(Entity):
    """
    Represents the player character in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the player can have a reference to it.
    x: int
        The x position of the player in the game map.
    y: int
        The y position of the player in the game map.
    player_type: PlayerType
        The constant data about this specific player.

    Attributes
    ----------
    melee_shader: MeleeShader
        The OpenGL shader used to find and attack any enemies within a specific distance
        around the player based on their direction.
    inventory_obj: Inventory
        The inventory object which represents this player's inventory.
    """

    # Class variables
    ID: EntityID = EntityID.PLAYER

    def __init__(self, game: Game, x: int, y: int, player_type: PlayerType) -> None:
        super().__init__(game, x, y, player_type)
        self.melee_shader: MeleeShader = MeleeShader(self.game)
        self.inventory_obj: Inventory = Inventory(self)

    def __repr__(self) -> str:
        return f"<Player (Position=({self.center_x}, {self.center_y}))>"

    @property
    def inventory(self) -> list[Item]:
        """Returns the player's inventory."""
        return self.inventory_obj.array

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

    def run_melee_shader(self) -> None:
        """Runs the melee shader to get all enemies within melee range of the player."""
        # Deal melee damage to any entity that the player can attack. This is determined
        # by the melee shader
        self.melee_attack(self.melee_shader.run_shader())
