from __future__ import annotations

# Builtin
from typing import Optional

# Pip
import arcade
from constants import FLOOR, WALL
from entities.tiles import Tile

# Custom
from generation.map import Map


class Game(arcade.Window):
    """
    Manages the game and its actions.

    Attributes
    ----------
    game_map: Optional[Map]
        The game map for the current level.
    game_sprite_list: Optional[arcade.SpriteList]
        The sprite list for the converted game map.
    """

    def __init__(self) -> None:
        super().__init__()
        self.game_map: Optional[Map] = None
        self.game_sprite_list: Optional[arcade.SpriteList] = None
        self.setup_level(1)
        arcade.set_background_color(arcade.color.WHITE)

    def setup_level(self, level: int) -> None:
        """
        Creates the game generation for a specific level.

        Parameters
        ----------
        level: int
            The level to create a generation for. Each level is more difficult than the
            previous.
        """
        self.game_map = Map(level)
        self.game_sprite_list = arcade.SpriteList(use_spatial_hash=True)
        for count_y, y in enumerate(self.game_map.grid):
            for count_x, x in enumerate(y):
                # Determine which type the tile is
                sprite = None
                if x == FLOOR:
                    sprite = Tile(count_x, count_y, FLOOR)
                elif x == WALL:
                    sprite = Tile(count_x, count_y, WALL)

                # Add the newly created sprite to the sprite list
                if sprite is not None:
                    self.game_sprite_list.append(sprite)

    def on_draw(self) -> None:
        """
        Render the screen.
        """
        arcade.start_render()

        # Draw the game map
        assert self.game_sprite_list is not None
        self.game_sprite_list.draw()


def main() -> None:
    """Initialises the game and runs it."""
    window = Game()
    window.run()


if __name__ == "__main__":
    main()
