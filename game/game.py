from __future__ import annotations

# Builtin
from typing import Optional

# Pip
import arcade
from constants import FLOOR, WALL
from entities.player import Player
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
        self.player: Optional[Player] = None
        self.camera: Optional[arcade.Camera] = None
        self.setup(1)
        arcade.set_background_color(arcade.color.BLACK)

    def setup(self, level: int) -> None:
        """
        Sets up the game.

        Parameters
        ----------
        level: int
            The level to create a generation for. Each level is more difficult than the
            previous.
        """
        # Create the game map
        self.game_map = Map(level)

        # Assign sprites to the game map
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

        # Create the player object
        self.player = Player(
            self.game_map.player_spawn[0], self.game_map.player_spawn[1]
        )

        # Set up the Camera
        self.camera = arcade.Camera(self.width, self.height)

    def on_draw(self) -> None:
        """Render the screen."""

        # Clear the screen
        arcade.start_render()

        # Activate our Camera
        assert self.camera is not None
        self.camera.use()

        # Draw the game map
        assert self.game_sprite_list is not None
        assert self.player is not None
        self.game_sprite_list.draw()
        self.player.draw()

    def on_update(self, delta_time: float) -> None:
        """
        Movement and game logic happens here.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Position the camera
        self.center_camera_on_player()

    def center_camera_on_player(self) -> None:
        """Centers the camera on the player."""
        # Calculate the screen position centered on the player
        assert self.player is not None
        assert self.camera is not None
        screen_center_x = self.player.center_x - (self.camera.viewport_width / 2)
        screen_center_y = self.player.center_y - (self.camera.viewport_height / 2)

        # Make sure the camera doesn't travel past 0
        if screen_center_x < 0:
            screen_center_x = 0
        if screen_center_y < 0:
            screen_center_y = 0

        # Move the camera to the new position
        new_position = (screen_center_x, screen_center_y)
        self.camera.move_to(new_position)  # noqa


def main() -> None:
    """Initialises the game and runs it."""
    window = Game()
    window.run()


if __name__ == "__main__":
    main()
