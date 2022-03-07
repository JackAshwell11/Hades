from __future__ import annotations

# Builtin
import math

# Pip
import arcade

# Custom
from constants import (
    ATTACK_COOLDOWN,
    DAMPING,
    ENEMY,
    ENEMY_HEALTH,
    ENEMY_VIEW_DISTANCE,
    FLOOR,
    PLAYER,
    PLAYER_HEALTH,
    PLAYER_MOVEMENT_FORCE,
    SPRITE_HEIGHT,
    SPRITE_WIDTH,
    WALL,
)
from entities.ai import FollowLineOfSight
from entities.enemy import Enemy
from entities.player import Player
from entities.tile import Tile
from generation.map import Map
from physics import PhysicsEngine
from textures import moving_textures, non_moving_textures, pos_to_pixel


class Game(arcade.View):
    """
    Manages the game and its actions.

    Parameters
    ----------
    debug_mode: bool
        Whether to draw the various debug things or not.

    Attributes
    ----------
    game_map: Map | None
        The game map for the current level.
    player: Player | None
        The sprite for the playable character in the game.
    floor_sprites: arcade.SpriteList
        The sprite list for the floor sprites.
    wall_sprites: arcade.SpriteList
        The sprite list for the wall sprites.
    enemies: arcade.SpriteList
        The sprite list for the enemy sprites.
    physics_engine: PhysicsEngine | None
        The physics engine which processes wall collision.
    camera: arcade.Camera | None
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

    def __init__(self, debug_mode: bool = False) -> None:
        super().__init__()
        self.debug_mode: bool = debug_mode
        self.game_map: Map | None = None
        self.player: Player | None = None
        self.floor_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.wall_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.bullet_sprites: arcade.SpriteList = arcade.SpriteList(
            use_spatial_hash=True
        )
        self.enemies: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.physics_engine: PhysicsEngine | None = None
        self.camera: arcade.Camera | None = None
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
            The level to create a generation for. Each level should be more difficult
            than the last.
        """
        # Create the game map
        self.game_map = Map(level)

        # Assign sprites to the game map
        for count_y, y in enumerate(self.game_map.grid):
            for count_x, x in enumerate(y):
                # Determine which type the tile is
                if x == FLOOR:
                    self.floor_sprites.append(
                        Tile(count_x, count_y, non_moving_textures["tiles"][0])
                    )
                elif x == WALL:
                    self.wall_sprites.append(
                        Tile(count_x, count_y, non_moving_textures["tiles"][1])
                    )
                elif x == PLAYER:
                    self.player = Player(
                        count_x,
                        count_y,
                        moving_textures["player"],
                        PLAYER_HEALTH,
                    )
                    self.floor_sprites.append(
                        Tile(count_x, count_y, non_moving_textures["tiles"][0])
                    )
                elif x == ENEMY:
                    self.enemies.append(
                        Enemy(
                            count_x,
                            count_y,
                            moving_textures["enemy"],
                            ENEMY_HEALTH,
                            FollowLineOfSight(),
                        )
                    )
                    self.floor_sprites.append(
                        Tile(count_x, count_y, non_moving_textures["tiles"][0])
                    )

        # Create the physics engine
        self.physics_engine = PhysicsEngine(DAMPING)
        self.physics_engine.setup(self.player, self.wall_sprites, self.enemies)

        # Set up the Camera
        self.camera = arcade.Camera(self.window.width, self.window.height)

    def on_show(self) -> None:
        """Called when the view loads."""
        # Set the background color
        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self) -> None:
        """Render the screen."""
        # Make sure variables needed are valid
        assert self.camera is not None
        assert self.player is not None

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

        # Draw the debug items
        if self.debug_mode:
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
        assert self.physics_engine is not None
        assert self.player is not None

        # Process logic for the enemies
        self.enemies.on_update()

        # Process logic for the player
        self.player.on_update()

        # Calculate the speed and direction of the player based on the keys pressed
        if self.up_pressed and not self.down_pressed:
            self.physics_engine.apply_force(self.player, (0, PLAYER_MOVEMENT_FORCE))
        elif self.down_pressed and not self.up_pressed:
            self.physics_engine.apply_force(self.player, (0, -PLAYER_MOVEMENT_FORCE))
        if self.left_pressed and not self.right_pressed:
            self.physics_engine.apply_force(self.player, (-PLAYER_MOVEMENT_FORCE, 0))
        elif self.right_pressed and not self.left_pressed:
            self.physics_engine.apply_force(self.player, (PLAYER_MOVEMENT_FORCE, 0))

        # Position the camera
        self.center_camera_on_player()

        # Update the physics engine
        self.physics_engine.step()

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
            and self.player.time_since_last_attack >= ATTACK_COOLDOWN
        ):
            # self.player.melee_attack(self.enemies, Damage.PLAYER)
            self.player.ranged_attack(self.bullet_sprites)

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float) -> None:
        """
        Called when the mouse moves.

        Parameters
        ----------
        x: float
            The x position of the mouse.
        y: float
            The y position of the mouse.
        dx: float
            The change in the x position.
        dy: float
            The change in the y position.
        """
        # Make sure variables needed are valid
        assert self.player is not None
        assert self.camera is not None

        # Calculate the new angle in degrees
        camera_x, camera_y = self.camera.position
        vec_x, vec_y = (
            x - self.player.center_x + camera_x,
            y - self.player.center_y + camera_y,
        )
        angle = math.degrees(math.atan2(vec_y, vec_x))
        if angle < 0:
            angle = angle + 360
        self.player.direction = angle
        self.player.facing = 1 if 90 <= angle <= 270 else 0

    def center_camera_on_player(self) -> None:
        """Centers the camera on the player."""
        # Make sure variables needed are valid
        assert self.camera is not None
        assert self.game_map is not None
        assert self.player is not None

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
