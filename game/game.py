from __future__ import annotations

# Builtin
from typing import Optional

# Pip
import arcade

# Custom
from constants import (
    ENEMY,
    FLOOR,
    MOVEMENT_SPEED,
    PLAYER,
    SPRITE_HEIGHT,
    SPRITE_WIDTH,
    WALL,
)
from entities.enemy import Enemy
from entities.entity import Entity, TileType
from entities.player import Player
from entities.tiles import Tile
from generation.map import Map
from textures.textures import pos_to_pixel


class Game(arcade.Window):
    """
    Manages the game and its actions.

    Attributes
    ----------
    game_map: Optional[Map]
        The game map for the current level.
    floor_sprites: arcade.SpriteList
        The sprite list for the floor sprites.
    wall_sprites: arcade.SpriteList
        The sprite list for the wall sprites.
    player: Optional[Player]
        The playable character in the game.
    enemies: arcade.SpriteList
        The sprite list for the enemy sprites.
    physics_engine: Optional[arcade.PhysicsEngineSimple]
        The physics engine which processes wall collision.
    camera: Optional[arcade.Camera]
        The camera used for moving the viewport around the screen.
    left_pressed: bool
        Whether the left key is pressed or not.
    right_pressed: bool
        Whether the right key is pressed or not.
    up_pressed: bool
        Whether the up key is pressed or not.
    down_pressed: bool
        Whether the down key is pressed or not.
    """

    def __init__(self) -> None:
        super().__init__()
        self.game_map: Optional[Map] = None
        self.floor_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.wall_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.player: Optional[Entity] = None
        self.enemies: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.physics_engine: Optional[arcade.PhysicsEngineSimple] = None
        self.camera: Optional[arcade.Camera] = None
        self.left_pressed: bool = False
        self.right_pressed: bool = False
        self.up_pressed: bool = False
        self.down_pressed: bool = False
        self.setup(1)

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
        self.floor_sprites = arcade.SpriteList(use_spatial_hash=True)
        self.wall_sprites = arcade.SpriteList(use_spatial_hash=True)
        for count_y, y in enumerate(self.game_map.grid):
            for count_x, x in enumerate(y):
                # Determine which type the tile is
                if x == FLOOR:
                    self.floor_sprites.append(
                        Entity(count_x, count_y, TileType.FLOOR, tile=Tile())
                    )
                elif x == WALL:
                    self.wall_sprites.append(
                        Entity(count_x, count_y, TileType.WALL, tile=Tile())
                    )
                elif x == PLAYER:
                    self.player = Entity(
                        count_x, count_y, TileType.PLAYER, character=Player()
                    )
                    self.floor_sprites.append(
                        Entity(count_x, count_y, TileType.FLOOR, tile=Tile())
                    )
                elif x == ENEMY:
                    self.enemies.append(
                        Entity(count_x, count_y, TileType.ENEMY, character=Enemy())
                    )
                    self.floor_sprites.append(
                        Entity(count_x, count_y, TileType.FLOOR, tile=Tile())
                    )

        # Create the physics engine
        self.physics_engine = arcade.PhysicsEngineSimple(self.player, self.wall_sprites)

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
        assert self.floor_sprites is not None
        assert self.wall_sprites is not None
        assert self.player is not None
        assert self.enemies is not None
        self.floor_sprites.draw(pixelated=True)
        self.wall_sprites.draw(pixelated=True)
        self.player.draw(pixelated=True)
        self.enemies.draw(pixelated=True)

    def on_update(self, delta_time: float) -> None:
        """
        Processes movement and game logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Position the camera
        self.center_camera_on_player()

        # Calculate the speed and direction of the player based on the keys pressed
        assert self.player is not None
        self.player.change_x, self.player.change_y = 0, 0

        if self.up_pressed and not self.down_pressed:
            self.player.change_y = MOVEMENT_SPEED
        elif self.down_pressed and not self.up_pressed:
            self.player.change_y = -MOVEMENT_SPEED
        if self.left_pressed and not self.right_pressed:
            self.player.change_x = -MOVEMENT_SPEED
        elif self.right_pressed and not self.left_pressed:
            self.player.change_x = MOVEMENT_SPEED

        # Update the physics engine which processes collision
        assert self.physics_engine is not None
        self.physics_engine.update()

    def on_key_press(self, key: int, modifiers: int) -> None:
        """
        Called when the player presses a key.

        Parameters
        ----------
        key: int
            The key that was hit.
        modifiers: int
            Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this
            event.
        """
        if key is arcade.key.W:
            self.up_pressed = True
        elif key is arcade.key.S:
            self.down_pressed = True
        elif key is arcade.key.A:
            self.left_pressed = True
        elif key is arcade.key.D:
            self.right_pressed = True

    def on_key_release(self, key: int, modifiers: int) -> None:
        """
        Called when the player releases a key.

        Parameters
        ----------
        key: int
            The key that was hit.
        modifiers: int
            Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this
            event.
        """
        if key is arcade.key.W:
            self.up_pressed = False
        elif key is arcade.key.S:
            self.down_pressed = False
        elif key is arcade.key.A:
            self.left_pressed = False
        elif key is arcade.key.D:
            self.right_pressed = False

    def center_camera_on_player(self) -> None:
        """Centers the camera on the player."""
        # Calculate the screen position centered on the player
        assert self.player is not None
        assert self.camera is not None
        screen_center_x = self.player.center_x - (self.camera.viewport_width / 2)
        screen_center_y = self.player.center_y - (self.camera.viewport_height / 2)

        # Calculate the maximum width and height a sprite can be
        assert self.game_map is not None
        upper_x, upper_y = pos_to_pixel(
            len(self.game_map.grid[0]) - 1, len(self.game_map.grid) - 1
        )

        # Calculate the maximum width and height the camera can be
        upper_camera_x, upper_camera_y = (
            upper_x
            - self.camera.viewport_width
            + (self.camera.viewport_width / SPRITE_WIDTH),
            upper_y
            - self.camera.viewport_height
            + (self.camera.viewport_height / SPRITE_HEIGHT),
        )

        # Make sure the camera doesn't extend beyond the boundaries
        if screen_center_x < 0:
            screen_center_x = 0
        elif screen_center_x > upper_camera_x:
            screen_center_x = upper_camera_x
        if screen_center_y < 0:
            screen_center_y = 0
        elif screen_center_y > upper_camera_y:
            screen_center_y = upper_camera_y

        # Move the camera to the new position
        self.camera.move_to((screen_center_x, screen_center_y))  # noqa


def main() -> None:
    """Initialises the game and runs it."""
    window = Game()
    window.run()


if __name__ == "__main__":
    main()
