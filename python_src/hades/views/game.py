"""Initialises and manages the main game."""
from __future__ import annotations

# Builtin
import logging
import math
import random
from functools import cache
from typing import NamedTuple

# Pip
import arcade
import pyglet.math
from hades_extensions import TileType, create_map

# Custom
from hades.constants import (
    CONSUMABLE_LEVEL_MAX_RANGE,
    DAMPING,
    DEBUG_ENEMY_SPAWN_COLOR,
    DEBUG_ENEMY_SPAWN_SIZE,
    DEBUG_GAME,
    ENEMY_GENERATE_INTERVAL,
    ENEMY_RETRY_COUNT,
    FACING_LEFT,
    FACING_RIGHT,
    LEVEL_GENERATOR_INTERVAL,
    SPRITE_SIZE,
    TOTAL_ENEMY_COUNT,
    GameObjectType,
)
from hades.game_objects.system import ECS
from hades.physics import PhysicsEngine
from hades.textures import grid_pos_to_pixel

__all__ = ("Game",)

# Get the logger
logger = logging.getLogger(__name__)


@cache
def get_upper_bound(level: int) -> int:
    """Get the upper bound for a given level.

    Args:
        level: The level to get the upper bound for.

    Returns:
        The upper bound.
    """
    return (level // LEVEL_GENERATOR_INTERVAL) + 1


class LevelConstants(NamedTuple):
    """Holds the constants for a specific level.

    level: int
        The level of this game.
    width: int
        The width of the game map.
    height: int
        The height of the game map.
    """

    level: int
    width: int
    height: int


class Game(arcade.View):
    """Manages the game and its actions.

    Attributes:
        level_constants: Holds the constants for the current level.
        system: The entity component system which manages the game objects.
        tile_sprites: The sprite list for the tile game objects.
        entity_sprites: The sprite list for the entity game objects.
        game_camera: The camera used for moving the viewport around the screen.
        gui_camera: The camera used for visualising the GUI elements.
        physics_engine: The physics engine which processes wall collision.
        possible_enemy_spawns: A list of possible positions that enemies can spawn in.
        player_status_text: The text object used for displaying the player's health and
            armour.
    """

    def __init__(self: Game, level: int) -> None:
        """Initialise the object.

        Args:
            level: The level to create a game for.
        """
        super().__init__()
        generation_result = create_map(level)
        self.level_constants: LevelConstants = LevelConstants(*generation_result[1])
        self.system: ECS = ECS()
        self.tile_sprites: arcade.SpriteList = arcade.SpriteList()
        self.entity_sprites: arcade.SpriteList = arcade.SpriteList()
        self.game_camera: arcade.Camera = arcade.Camera()
        self.gui_camera: arcade.Camera = arcade.Camera()
        self.physics_engine: PhysicsEngine = PhysicsEngine(DAMPING)
        self.possible_enemy_spawns: list[tuple[int, int]] = []
        self.player_status_text: arcade.Text = arcade.Text(
            "Money: 0",
            10,
            10,
            font_size=20,
        )

        # Initialise the game objects
        for count, tile in enumerate(generation_result[0]):
            # Get the screen position from the grid position
            position = (
                count % self.level_constants.width,
                count // self.level_constants.width,
            )

            # Determine the type of the tile
            if tile == TileType.Wall:
                self.tile_sprites.append(
                    self.system.add_game_object(GameObjectType.WALL, position),
                )
                continue
            if tile == TileType.Player:
                self.entity_sprites.append(
                    self.system.add_game_object(GameObjectType.PLAYER, position),
                )
            # TODO: Have converter for TileType to GameObjectType (or modify TileType to have Enemy item)
            else:
                self.tile_sprites.append(
                    self.system.add_game_object(GameObjectType.HEALTH_POTION, position),
                )
            # self._initialise_game_object(
            #     CONSUMABLES[tile],  # noqa: ERA001
            #     self.tile_sprites,  # noqa: ERA001
            #     position,  # noqa: ERA001

            # Make the tile's backdrop a floor
            self.tile_sprites.append(
                self.system.add_game_object(GameObjectType.FLOOR, position),
            )

        # Generate half of the total enemies allowed then schedule their generation
        for _ in range(TOTAL_ENEMY_COUNT // 2):
            self.generate_enemy()
        arcade.schedule(
            self.generate_enemy,
            ENEMY_GENERATE_INTERVAL,
        )

    def on_draw(self: Game) -> None:
        """Render the screen."""
        # Clear the screen and set the background color
        self.clear()
        self.window.background_color = arcade.color.BLACK

        # Activate our game camera
        self.game_camera.use()

        # Draw the various spritelists
        self.tile_sprites.draw(pixelated=True)
        self.entity_sprites.draw(pixelated=True)

        # Draw the stuff needed for the debug mode
        if DEBUG_GAME:
            # Draw the enemy spawn locations
            arcade.draw_points(
                [
                    grid_pos_to_pixel(*location)
                    for location in self.possible_enemy_spawns
                ],
                DEBUG_ENEMY_SPAWN_COLOR,
                DEBUG_ENEMY_SPAWN_SIZE,
            )

        # Draw the gui on the screen
        self.gui_camera.use()
        self.player_status_text.value = f"Money: {str(self.player.money.value)}"
        self.player_status_text.draw()

    def on_update(self: Game, delta_time: float) -> None:
        """Process movement and game logic.

        Args:
            delta_time: Time interval since the last time the function was called.
        """
        # Check if the game should end
        # if self.player.health.value <= 0 or not self.enemy_sprites:

        # Process logic for the player
        self.player.on_update(delta_time)

        # Process logic for the enemies
        self.enemy_sprites.on_update(delta_time)

        # Process logic for the bullets
        self.bullet_sprites.on_update(delta_time)

        # Update the physics engine
        self.physics_engine.step()

        # Position the camera
        self.center_camera_on_player()

    def on_key_press(self: Game, symbol: int, modifiers: int) -> None:
        """Process key press functionality.

        Args:
            symbol: The key that was hit.
            modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        logger.debug(
            "Received key press with key %r and modifiers %r",
            symbol,
            modifiers,
        )
        match symbol:
            case arcade.key.W:
                self.player.up_pressed = True
            case arcade.key.S:
                self.player.down_pressed = True
            case arcade.key.A:
                self.player.left_pressed = True
            case arcade.key.D:
                self.player.right_pressed = True

    def on_key_release(self: Game, key: int, modifiers: int) -> None:
        """Process key release functionality.

        Args:
            key: The key that was hit.
            modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        logger.debug(
            "Received key release with key %r and modifiers %r",
            key,
            modifiers,
        )
        match key:
            case arcade.key.W:
                self.player.up_pressed = False
            case arcade.key.S:
                self.player.down_pressed = False
            case arcade.key.A:
                self.player.left_pressed = False
            case arcade.key.D:
                self.player.right_pressed = False

    def on_mouse_press(self: Game, x: int, y: int, button: int, modifiers: int) -> None:
        """Process mouse button functionality.

        Args:
            x: The x position of the mouse.
            y: The y position of the mouse.
            button: Which button was hit.
            modifiers:Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        logger.debug(
            "%r mouse button was pressed at position (%f, %f) with modifiers %r",
            button,
            x,
            y,
            modifiers,
        )
        if button is arcade.MOUSE_BUTTON_LEFT:
            self.player.attack()

    def on_mouse_motion(self: Game, x: int, y: int, *_: int) -> None:
        """Process mouse motion functionality.

        Args:
            x: The x position of the mouse.
            y: The y position of the mouse.
        """
        # Calculate the new angle in degrees
        camera_x, camera_y = self.game_camera.position
        vec_x, vec_y = (
            x - self.player.center_x + camera_x,
            y - self.player.center_y + camera_y,
        )
        angle = math.degrees(math.atan2(vec_y, vec_x))
        if angle < 0:
            angle += 360
        self.player.direction = angle
        self.player.facing = FACING_LEFT if 90 <= angle <= 270 else FACING_RIGHT

    def generate_enemy(self: Game, _: float = 1 / 60) -> None:
        """Generate an enemy outside the player's fov."""
        # Check if we've reached the max amount of enemies
        if len(self.enemy_sprites) < TOTAL_ENEMY_COUNT:
            # Limit not reached so determine the bounds
            enemy_upper_bound = get_upper_bound(self.level_constants.level)
            enemy_lower_bound = (
                1
                if enemy_upper_bound - 1 < CONSUMABLE_LEVEL_MAX_RANGE
                else enemy_upper_bound - CONSUMABLE_LEVEL_MAX_RANGE
            )
            logger.debug(
                "Generating enemy with lower bound %d and upper bound %d",
                enemy_lower_bound,
                enemy_upper_bound,
            )

            # Attempt to generate an enemy retrying ENEMY_RETRY_COUNT times if it fails
            enemy = Enemy(
                self,
                0,
                0,
                ENEMY1,
                min(
                    random.randint(enemy_lower_bound, enemy_upper_bound),
                    ENEMY1.entity_data.level_limit,
                ),
            )
            tries = ENEMY_RETRY_COUNT
            while tries != 0:
                # Pick a random position for the enemy and check if they collide with
                # other enemies or the player
                enemy.position = grid_pos_to_pixel(*self.possible_enemy_spawns.pop())
                if enemy.collides_with_list(
                    self.enemy_sprites,
                ) or enemy.collides_with_sprite(self.player):
                    logger.debug(
                        "%r encountered a collision during generation. Retrying",
                        enemy,
                    )
                    tries -= 1
                    continue

                # Enemy position is good so add them to the spritelist and stop
                self.physics_engine.add_enemy(enemy)
                self.enemy_sprites.append(enemy)
                logger.debug("%d has been successfully generated")
                break

            # Enemy has failed to generate
            logger.debug("%r failed to be generated", enemy)

    def center_camera_on_player(self: Game) -> None:
        """Centers the camera on the player."""
        # Calculate the screen position centered on the player
        screen_center_x = self.player.center_x - (self.game_camera.viewport_width / 2)
        screen_center_y = self.player.center_y - (self.game_camera.viewport_height / 2)

        # Calculate the maximum width and height of the game map
        upper_x, upper_y = grid_pos_to_pixel(
            self.level_constants.width,
            self.level_constants.height,
        )

        # Calculate the maximum width and height the camera can be
        half_sprite_size = SPRITE_SIZE / 2
        upper_camera_x, upper_camera_y = (
            upper_x - self.game_camera.viewport_width - half_sprite_size,
            upper_y - self.game_camera.viewport_height - half_sprite_size,
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
        new_position = pyglet.math.Vec2(screen_center_x, screen_center_y)

        # Check if the camera position has changed
        if self.game_camera.position != new_position:
            # Move the camera to the new position
            self.game_camera.move_to(new_position)

    def __repr__(self: Game) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Game (Current window={self.window})>"
