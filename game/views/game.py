from __future__ import annotations

# Builtin
import math
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants.entity import (
    ATTACK_COOLDOWN,
    ENEMY1,
    FACING_LEFT,
    FACING_RIGHT,
    MOVEMENT_FORCE,
    PLAYER,
)
from constants.general import (
    DAMPING,
    DEBUG_ATTACK_DISTANCE,
    DEBUG_VIEW_DISTANCE,
    SPRITE_SIZE,
)
from constants.generation import TileType
from entities.ai import FollowLineOfSight
from entities.enemy import Enemy
from entities.item import HealthPotion, Shop
from entities.player import Player
from entities.tile import Floor, Wall
from generation.map import Map
from physics import PhysicsEngine
from textures import pos_to_pixel
from views.inventory_view import InventoryView

if TYPE_CHECKING:
    from entities.base import Item


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
    wall_sprites: arcade.SpriteList
        The sprite list for the wall sprites. This is only used for updating the melee
        shader.
    tile_sprites: arcade.SpriteList
        The sprite list for the tile sprites. This is used for drawing the different
        tiles.
    item_sprites: arcade.SpriteList
        The sprite list for the item sprites. This is only used for detecting player
        activity around the item.
    bullet_sprites: arcade.SpriteList
        The sprite list for the bullet sprites.
    enemies: arcade.SpriteList
        The sprite list for the enemy sprites.
    physics_engine: PhysicsEngine | None
        The physics engine which processes wall collision.
    camera: arcade.Camera | None
        The camera used for moving the viewport around the screen.
    gui_camera: arcade.Camera | None
        The camera used for visualising the GUI elements.
    health_text: arcade.Text
        The text object used for displaying the player's health.
    nearest_item: Item | None
        Stores the nearest item so the player can activate it.
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
        self.wall_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.tile_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.item_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.bullet_sprites: arcade.SpriteList = arcade.SpriteList()
        self.enemies: arcade.SpriteList = arcade.SpriteList()
        self.physics_engine: PhysicsEngine | None = None
        self.camera: arcade.Camera | None = None
        self.gui_camera: arcade.Camera | None = None
        self.health_text: arcade.Text = arcade.Text(
            "Score: 0  Health: 0",
            10,
            10,
            arcade.color.WHITE,
            20,
        )
        self.nearest_item: Item | None = None
        self.item_text: arcade.Text = arcade.Text(
            "Press E to activate",
            self.window.width / 2 - 150,
            self.window.height / 2 - 200,
            arcade.color.BLACK,
            20,
        )
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
                match x:
                    case TileType.FLOOR.value:
                        self.tile_sprites.append(Floor(count_x, count_y))
                    case TileType.WALL.value:
                        wall = Wall(count_x, count_y)
                        self.wall_sprites.append(wall)
                        self.tile_sprites.append(wall)
                    case TileType.PLAYER.value:
                        self.player = Player(
                            self,
                            count_x,
                            count_y,
                            PLAYER,
                        )
                        self.tile_sprites.append(Floor(count_x, count_y))
                    case TileType.ENEMY.value:
                        self.enemies.append(
                            Enemy(
                                self,
                                count_x,
                                count_y,
                                ENEMY1,
                                FollowLineOfSight(),
                            )
                        )
                        self.tile_sprites.append(Floor(count_x, count_y))
                    case TileType.HEALTH_POTION.value:
                        self.tile_sprites.append(Floor(count_x, count_y))
                        health_potion = HealthPotion(self, count_x, count_y)
                        self.tile_sprites.append(health_potion)
                        self.item_sprites.append(health_potion)
                    case TileType.SHOP.value:
                        self.tile_sprites.append(Floor(count_x, count_y))
                        shop = Shop(self, count_x, count_y)
                        self.tile_sprites.append(shop)
                        self.item_sprites.append(shop)

        # Make sure the player was actually created
        assert self.player is not None

        # Create the physics engine
        self.physics_engine = PhysicsEngine(DAMPING)
        self.physics_engine.setup(self.player, self.tile_sprites, self.enemies)

        # Set up the Camera
        self.camera = arcade.Camera(self.window.width, self.window.height)
        self.gui_camera = arcade.Camera(self.window.width, self.window.height)

        # Set up the melee shader
        self.player.melee_shader.setup_shader()

        # Check if any enemy has line of sight
        for enemy in self.enemies:
            enemy.check_line_of_sight()  # noqa

        # Set up the inventory view
        inventory_view = InventoryView(self.player)
        self.window.views["InventoryView"] = inventory_view

    def on_show(self) -> None:
        """Called when the view loads."""
        # Set the background color
        self.window.background_color = arcade.color.BLACK

    def on_draw(self) -> None:
        """Render the screen."""
        # Make sure variables needed are valid
        assert self.camera is not None
        assert self.player is not None
        assert self.gui_camera is not None

        # Clear the screen
        self.clear()

        # Activate our Camera
        self.camera.use()

        # Draw the game map
        self.tile_sprites.draw(pixelated=True)
        self.bullet_sprites.draw(pixelated=True)
        self.enemies.draw(pixelated=True)
        self.player.draw(pixelated=True)

        # Draw the debug items
        if self.debug_mode:
            for enemy in self.enemies:
                # Draw the enemy's view distance
                arcade.draw_circle_outline(
                    enemy.center_x,
                    enemy.center_y,
                    enemy.entity_type.view_distance * SPRITE_SIZE,  # noqa
                    DEBUG_VIEW_DISTANCE,
                )
                # Draw the enemy's attack distance
                arcade.draw_circle_outline(
                    enemy.center_x,
                    enemy.center_y,
                    enemy.entity_type.attack_range * SPRITE_SIZE,  # noqa
                    DEBUG_ATTACK_DISTANCE,
                )

            # Calculate the two boundary points for the player fov
            half_angle = self.player.entity_type.melee_degree // 2
            low_angle = math.radians(self.player.direction - half_angle)
            high_angle = math.radians(self.player.direction + half_angle)
            point_low = (
                self.player.center_x
                + math.cos(low_angle)
                * SPRITE_SIZE
                * self.player.entity_type.melee_range,
                self.player.center_y
                + math.sin(low_angle)
                * SPRITE_SIZE
                * self.player.entity_type.melee_range,
            )
            point_high = (
                self.player.center_x
                + math.cos(high_angle)
                * SPRITE_SIZE
                * self.player.entity_type.melee_range,
                self.player.center_y
                + math.sin(high_angle)
                * SPRITE_SIZE
                * self.player.entity_type.melee_range,
            )
            # Draw both boundary lines for the player fov
            arcade.draw_line(
                self.player.center_x,
                self.player.center_y,
                *point_low,
                DEBUG_ATTACK_DISTANCE,
            )
            arcade.draw_line(
                self.player.center_x,
                self.player.center_y,
                *point_high,
                DEBUG_ATTACK_DISTANCE,
            )
            # Draw the arc between the two making sure to double the width and height
            # since the radius is calculated, but we want the diameter
            arcade.draw_arc_outline(
                self.player.center_x,
                self.player.center_y,
                math.hypot(point_high[0] - point_low[0], point_high[1] - point_low[1])
                * 2,
                self.player.entity_type.melee_range * SPRITE_SIZE * 2,
                DEBUG_ATTACK_DISTANCE,
                math.degrees(low_angle),
                math.degrees(high_angle),
                2,
            )

        # Draw the gui on the screen
        self.gui_camera.use()
        self.health_text.value = f"Health: {self.player.health}"
        self.health_text.draw()
        if self.nearest_item:
            self.item_text.draw()

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

        # Check if the game should end
        if self.player.health <= 0 or not self.enemies:
            arcade.exit()

        # Calculate the vertical velocity of the player based on the keys pressed
        update_enemies = False
        vertical_force = None
        if self.up_pressed and not self.down_pressed:
            vertical_force = (0, MOVEMENT_FORCE)
        elif self.down_pressed and not self.up_pressed:
            vertical_force = (0, -MOVEMENT_FORCE)
        if vertical_force:
            # Apply the vertical force
            self.physics_engine.apply_force(self.player, vertical_force)

            # Set update_enemies
            update_enemies = True

        # Calculate the horizontal velocity of the player based on the keys pressed
        horizontal_force = None
        if self.left_pressed and not self.right_pressed:
            horizontal_force = (-MOVEMENT_FORCE, 0)
        elif self.right_pressed and not self.left_pressed:
            horizontal_force = (MOVEMENT_FORCE, 0)
        if horizontal_force:
            # Apply the horizontal force
            self.physics_engine.apply_force(self.player, horizontal_force)

            # Set update_enemies
            update_enemies = True

        # Check if we need to update the enemy's line of sight
        if update_enemies:
            # Update the enemy's line of sight check
            for enemy in self.enemies:
                enemy.check_line_of_sight()  # noqa

        # Position the camera
        self.center_camera_on_player()

        # Process logic for the enemies
        self.enemies.on_update()

        # Process logic for the player
        self.player.on_update()

        # Update the physics engine
        self.physics_engine.step()

        # Check for any nearby items
        item_collision = arcade.check_for_collision_with_list(
            self.player, self.item_sprites
        )
        if item_collision:
            # Set nearest_item since we are colliding with an item
            self.nearest_item = item_collision[0]
        else:
            # Reset nearest_item since we don't want to activate an item that the player
            # is not colliding with
            self.nearest_item = None

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
        # Make sure variables needed are valid
        assert self.player is not None

        # Find out what key was pressed
        match key:
            case arcade.key.W:
                self.up_pressed = True
            case arcade.key.S:
                self.down_pressed = True
            case arcade.key.A:
                self.left_pressed = True
            case arcade.key.D:
                self.right_pressed = True
            case arcade.key.E:
                if self.nearest_item:
                    self.nearest_item.item_activate()
            case arcade.key.R:
                if self.nearest_item:
                    self.nearest_item.item_use()
            case arcade.key.F:
                self.window.show_view(self.window.views["InventoryView"])
                self.window.views["InventoryView"].manager.enable()

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
        match key:
            case arcade.key.W:
                self.up_pressed = False
            case arcade.key.S:
                self.down_pressed = False
            case arcade.key.A:
                self.left_pressed = False
            case arcade.key.D:
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
            # Reset the player's counter
            self.player.time_since_last_attack = 0

            # Attack
            self.player.ranged_attack(self.bullet_sprites)
            # self.player.run_melee_shader()

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
            angle += 360
        self.player.direction = angle
        self.player.facing = FACING_LEFT if 90 <= angle <= 270 else FACING_RIGHT

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
            + (self.camera.viewport_width / SPRITE_SIZE),
            upper_y
            - self.camera.viewport_height
            + (self.camera.viewport_height / SPRITE_SIZE),
        )

        # Store the old position, so we can check if it has changed
        old_position = (self.camera.position[0], self.camera.position[1])

        # Make sure the camera doesn't extend beyond the boundaries
        if screen_center_x < 0:
            screen_center_x = 0
        elif screen_center_x > upper_camera_x:
            screen_center_x = upper_camera_x
        if screen_center_y < 0:
            screen_center_y = 0
        elif screen_center_y > upper_camera_y:
            screen_center_y = upper_camera_y
        new_position = screen_center_x, screen_center_y

        # Check if the camera position has changed
        if old_position != new_position:
            # Move the camera to the new position
            self.camera.move_to((screen_center_x, screen_center_y))  # noqa
            # Update the melee shader collision framebuffer
            self.player.melee_shader.update_collision()
