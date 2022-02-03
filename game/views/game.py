from __future__ import annotations

# Builtin
from typing import Optional

# Pip
import arcade

# Custom
from constants import (
    ATTACK_COOLDOWN,
    DAMPING,
    ENEMY,
    ENEMY_VIEW_DISTANCE,
    FLOOR,
    PLAYER,
    PLAYER_MOVEMENT_FORCE,
    SPRITE_HEIGHT,
    SPRITE_WIDTH,
    WALL,
)
from entities.ai import FollowLineOfSight
from entities.character import Character
from entities.entity import Entity, TileType
from generation.map import Map
from physics import PhysicsEngine
from textures.textures import pos_to_pixel


class Game(arcade.View):
    """
    Manages the game and its actions.

    Parameters
    ----------
    draw_debug_circles: bool
        Whether to draw the view distance as a circle around each enemy/player or not.

    Attributes
    ----------
    game_map: Optional[Map]
        The game map for the current level.
    floor_sprites: arcade.SpriteList
        The sprite list for the floor sprites.
    wall_sprites: arcade.SpriteList
        The sprite list for the wall sprites.
    player: Optional[Entity]
        The playable character in the game.
    enemies: arcade.SpriteList
        The sprite list for the enemy sprites.
    physics_engine: Optional[arcade.PymunkPhysicsEngine]
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

    def __init__(self, draw_debug_circles: bool = False) -> None:
        super().__init__()
        self.draw_debug_circles: bool = draw_debug_circles
        self.game_map: Optional[Map] = None
        self.floor_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.wall_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.bullet_sprites: arcade.SpriteList = arcade.SpriteList(
            use_spatial_hash=True
        )
        self.player: Optional[Entity] = None
        self.enemies: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.physics_engine: Optional[arcade.PymunkPhysicsEngine] = None
        self.camera: Optional[arcade.Camera] = None
        self.left_pressed: bool = False
        self.right_pressed: bool = False
        self.up_pressed: bool = False
        self.down_pressed: bool = False

    def __repr__(self) -> str:
        return f"<Game (Current window={self.window})>"

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
                        Entity(count_x, count_y, TileType.FLOOR, is_tile=True)
                    )
                elif x == WALL:
                    self.wall_sprites.append(
                        Entity(count_x, count_y, TileType.WALL, is_tile=True)
                    )
                elif x == PLAYER:
                    self.player = Entity(
                        count_x, count_y, TileType.PLAYER, character=Character()
                    )
                    self.floor_sprites.append(
                        Entity(count_x, count_y, TileType.FLOOR, is_tile=True)
                    )
                elif x == ENEMY:
                    self.enemies.append(
                        Entity(
                            count_x,
                            count_y,
                            TileType.ENEMY,
                            character=Character(ai=FollowLineOfSight()),
                        )
                    )
                    self.floor_sprites.append(
                        Entity(count_x, count_y, TileType.FLOOR, is_tile=True)
                    )

        # Create the physics engine
        self.physics_engine = PhysicsEngine(DAMPING)
        self.physics_engine.setup(self.player, self.wall_sprites, self.enemies)

        # Set up the Camera
        self.camera = arcade.Camera(self.window.width, self.window.height)

    def on_show(self) -> None:
        """Set up the screen so this view can work properly."""
        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self) -> None:
        """Render the screen."""
        # Make sure variables needed are valid
        assert self.camera is not None
        assert self.floor_sprites is not None
        assert self.wall_sprites is not None
        assert self.player is not None
        assert self.enemies is not None

        # Clear the screen
        self.clear()

        # Activate our Camera
        self.camera.use()

        # Draw the game map
        self.floor_sprites.draw(pixelated=True)
        self.wall_sprites.draw(pixelated=True)
        self.bullet_sprites.draw(pixelated=True)
        self.player.draw(pixelated=True)
        self.enemies.draw(pixelated=True)

        # Draw the circles if draw_enemy_debug_circles is set to true
        if self.draw_debug_circles:
            for enemy in self.enemies:
                arcade.draw_circle_outline(
                    enemy.center_x,
                    enemy.center_y,
                    ENEMY_VIEW_DISTANCE * SPRITE_WIDTH,
                    arcade.color.RED,
                )

    def on_update(self, delta_time: float) -> None:
        """
        Processes movement and game logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Make sure variables needed are valid
        assert self.player is not None
        assert self.physics_engine is not None
        assert self.enemies is not None

        # Update the character's time since last attack
        self.player.character.time_since_last_attack += delta_time
        for enemy in self.enemies:
            enemy.character.time_since_last_attack += delta_time

        # Calculate the speed and direction of the player based on the keys pressed
        self.player.change_x, self.player.change_y = 0, 0

        if self.up_pressed and not self.down_pressed:
            self.physics_engine.apply_force(self.player, (0, PLAYER_MOVEMENT_FORCE))
        elif self.down_pressed and not self.up_pressed:
            self.physics_engine.apply_force(self.player, (0, -PLAYER_MOVEMENT_FORCE))
        if self.left_pressed and not self.right_pressed:
            self.physics_engine.apply_force(self.player, (-PLAYER_MOVEMENT_FORCE, 0))
        elif self.right_pressed and not self.left_pressed:
            self.physics_engine.apply_force(self.player, (PLAYER_MOVEMENT_FORCE, 0))

        # Update the physics engine
        self.physics_engine.step()

        # Position the camera
        self.center_camera_on_player()

        # Move the enemies
        for enemy in self.enemies:
            force = enemy.character.ai.calculate_movement(
                self.player, self.wall_sprites
            )
            physics_obj = self.physics_engine.get_physics_object(enemy)
            physics_obj.body.apply_force_at_local_point(force, (0, 0))

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

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        """
        Called when the player presses the mouse button.

        Parameters
        ----------
        x: float
            The x position of the mouse.
        y: float
            The y position of the mouse.
        button: int
            Which button was hit.
        modifiers: int
            Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this
            event.
        """
        # Make sure variables needed are valid
        assert self.player is not None

        # Test if the player can attack
        if (
            button == arcade.MOUSE_BUTTON_LEFT
            and self.player.character.time_since_last_attack >= ATTACK_COOLDOWN
        ):
            self.player.character.ranged_attack(x, y, self.bullet_sprites)

    def center_camera_on_player(self) -> None:
        """Centers the camera on the player."""
        # Make sure variables needed are valid
        assert self.player is not None
        assert self.camera is not None
        assert self.game_map is not None

        # Calculate the screen position centered on the player
        screen_center_x = self.player.center_x - (self.camera.viewport_width / 2)
        screen_center_y = self.player.center_y - (self.camera.viewport_height / 2)

        # Calculate the maximum width and height a sprite can be
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
