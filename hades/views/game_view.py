"""Initialises and manages the main game."""
from __future__ import annotations

# Builtin
import contextlib
import logging
import math
import random
from functools import cache
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from hades.constants.constructor import CONSUMABLES, ENEMY1, PLAYERS
from hades.constants.game_object import FACING_LEFT, FACING_RIGHT, SPRITE_SIZE
from hades.constants.general import (
    CONSUMABLE_LEVEL_MAX_RANGE,
    DAMPING,
    DEBUG_ATTACK_DISTANCE,
    DEBUG_ENEMY_SPAWN_COLOR,
    DEBUG_ENEMY_SPAWN_SIZE,
    DEBUG_GAME,
    DEBUG_VECTOR_FIELD_LINE,
    DEBUG_VIEW_DISTANCE,
    ENEMY_GENERATE_INTERVAL,
    ENEMY_RETRY_COUNT,
    LEVEL_GENERATOR_INTERVAL,
    TOTAL_ENEMY_COUNT,
)
from hades.constants.generation import WALL_REPLACEABLE_TILES, TileType
from hades.game_objects.attack import AreaOfEffectAttack, MeleeAttack
from hades.game_objects.enemy import Enemy
from hades.game_objects.player import Player
from hades.game_objects.tile import Consumable, Floor, Wall
from hades.generation.map import create_map
from hades.physics import PhysicsEngine
from hades.textures import grid_pos_to_pixel
from hades.vector_field import VectorField
from hades.views.base_view import BaseView
from hades.views.inventory_view import InventoryView
from hades.views.shop_view import ShopView

if TYPE_CHECKING:
    from hades.game_objects.base import CollectibleTile, UsableTile
    from hades.generation.map import LevelConstants

__all__ = ("Game",)

# Get the logger
logger = logging.getLogger(__name__)


@cache
def get_upper_bound(level: int) -> int:
    """Get the upper bound for a given level.

    Parameters
    ----------
    level: int
        The level to get the upper bound for.

    Returns
    -------
    int
        The upper bound.
    """
    return (level // LEVEL_GENERATOR_INTERVAL) + 1


class Game(BaseView):
    """Manages the game and its actions.

    Attributes
    ----------
    level_constants: LevelConstants | None
        Holds the constants for the current level.
    player: Player | None
        The sprite for the playable character in the game.
    item_sprites: arcade.SpriteList
        The sprite list for the item sprites. This is only used for detecting player
        activity around the item.
    wall_sprites: arcade.SpriteList
        The sprite list for the wall sprites. This is only used for updating the melee
        shader.
    tile_sprites: arcade.SpriteList
        The sprite list for the tile sprites. This is used for drawing the different
        tiles.
    bullet_sprites: arcade.SpriteList
        The sprite list for the bullet sprites.
    enemy_sprites: arcade.SpriteList
        The sprite list for the enemy sprites.
    enemy_indicator_bar_sprites: arcade.SpriteList
        The sprite list for drawing the enemy indicator bars.
    player_gui_sprites: arcade.SpriteList
        The sprite list for drawing the player's GUI.
    game_camera: arcade.Camera
        The camera used for moving the viewport around the screen.
    gui_camera: arcade.Camera
        The camera used for visualising the GUI elements.
    vector_field: VectorField | None
        The vector field which allows for easy pathfinding for the enemy AI.
    physics_engine: PhysicsEngine | None
        The physics engine which processes wall collision.
    nearest_item: CollectibleTile | UsableTile | None
        Stores the nearest item so the player can activate it.
    possible_enemy_spawns: list[tuple[int, int]]
        A list of possible positions that enemies can spawn in.
    player_status_text: arcade.Text
        The text object used for displaying the player's health and armour.
    item_text: arcade.Text
        The text object used for displaying info about the nearest item.
    """

    def __init__(self) -> None:
        super().__init__()
        self.background_color = arcade.color.BLACK
        self.level_constants: LevelConstants | None = None
        self.player: Player | None = None
        self.item_sprites: arcade.SpriteList = arcade.SpriteList(use_spatial_hash=True)
        self.wall_sprites: arcade.SpriteList = arcade.SpriteList()
        self.tile_sprites: arcade.SpriteList = arcade.SpriteList()
        self.bullet_sprites: arcade.SpriteList = arcade.SpriteList()
        self.enemy_sprites: arcade.SpriteList = arcade.SpriteList()
        self.enemy_indicator_bar_sprites: arcade.SpriteList = arcade.SpriteList()
        self.player_gui_sprites: arcade.SpriteList = arcade.SpriteList()
        self.game_camera: arcade.Camera = arcade.Camera(
            self.window.width, self.window.height
        )
        self.gui_camera: arcade.Camera = arcade.Camera(
            self.window.width, self.window.height
        )
        self.vector_field: VectorField | None = None
        self.physics_engine: PhysicsEngine | None = None
        self.nearest_item: CollectibleTile | UsableTile | None = None
        self.possible_enemy_spawns: list[tuple[int, int]] = []
        self.player_status_text: arcade.Text = arcade.Text(
            "Money: 0",
            10,
            10,
            font_size=20,
        )
        self.item_text: arcade.Text = arcade.Text(
            "",
            self.window.width / 2 - 150,
            self.window.height / 2 - 200,
            arcade.color.BLACK,
            20,
        )

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<Game (Current window={self.window})>"

    def post_hide_view(self) -> None:
        """Process post hide view functionality."""
        # Make sure variables needed are valid
        assert self.player is not None

        # Stop the player from moving after the game view is shown again
        self.player.left_pressed = (
            self.player.right_pressed
        ) = self.player.up_pressed = self.player.down_pressed = False

    def setup(self, level: int) -> None:
        """Set up the game.

        Parameters
        ----------
        level: int
            The level to create a game for.
        """
        # Calculate the lower and upper bounds that will determine the consumable
        # levels
        consumable_upper_bound = get_upper_bound(level)
        consumable_lower_bound = (
            1
            if consumable_upper_bound - 1 < CONSUMABLE_LEVEL_MAX_RANGE
            else consumable_upper_bound - CONSUMABLE_LEVEL_MAX_RANGE
        )
        logger.debug(
            "Generating consumable with lower bound %d and upper bound %d",
            consumable_lower_bound,
            consumable_upper_bound,
        )

        # Create the game map
        game_map, self.level_constants = create_map(level)

        # Assign sprites to the game map and initialise the vector grid
        for count_y, y in enumerate(reversed(game_map.grid)):
            for count_x, x in enumerate(y):
                # Determine if the tile is empty
                if x in WALL_REPLACEABLE_TILES:
                    continue

                # Determine if the tile is a wall
                if x == TileType.WALL:
                    wall = Wall(self, count_x, count_y)
                    self.wall_sprites.append(wall)
                    self.tile_sprites.append(wall)
                    continue

                # The tile's backdrop should be a floor
                floor = Floor(self, count_x, count_y)
                self.tile_sprites.append(floor)

                # Skip to the next iteration if the tile is a floor
                if x == TileType.FLOOR:
                    continue

                # Determine if the tile is a player or a consumable
                if x in PLAYERS:
                    self.player = Player(self, count_x, count_y, PLAYERS[x])
                elif x in CONSUMABLES:
                    target_constructor = CONSUMABLES[x]
                    instantiated_consumable = Consumable(
                        self,
                        count_x,
                        count_y,
                        target_constructor,
                        min(
                            random.randint(
                                consumable_lower_bound, consumable_upper_bound
                            ),
                            target_constructor.level_limit,
                        ),
                    )
                    self.tile_sprites.append(instantiated_consumable)
                    self.item_sprites.append(instantiated_consumable)
                else:
                    # Unknown type
                    logger.warning("Unknown TileType %r", x)
                    continue

        # Make sure the game map shape was set and the player was actually created
        assert self.level_constants is not None
        assert self.player is not None

        # Debug how many enemies and tiles were initialised
        logger.debug("Initialised game view with  %d tiles", len(self.tile_sprites))

        # Set up the inventory view
        inventory_view = InventoryView(self.player)
        self.window.views["InventoryView"] = inventory_view
        logger.info("Initialised inventory view")

        # Set up the shop view
        shop_view = ShopView(self.player)
        self.window.views["ShopView"] = shop_view
        logger.info("Initialised shop view")

        # Create the physics engine
        self.physics_engine = PhysicsEngine(DAMPING)
        self.physics_engine.setup(self.player, self.tile_sprites)

        # Initialise the vector field
        self.vector_field = VectorField(
            self.level_constants.width,
            self.level_constants.height,
            self.wall_sprites,
        )
        self.possible_enemy_spawns = self.vector_field.recalculate_map(
            self.player.position, self.player.entity_data.view_distance
        )
        logger.debug(
            "Created vector grid with height %d and width %d",
            self.vector_field.height,
            self.vector_field.width,
        )

        # Update the player's current tile position
        self.player.current_tile_pos = self.vector_field.get_tile_pos_for_pixel(
            self.player.position
        )

        # Set up the melee shader
        # self.player.melee_shader.setup_shader()

        # Generate half of the total enemies and then schedule the function to run
        # every ENEMY_GENERATE_INTERVAL seconds
        for _ in range(TOTAL_ENEMY_COUNT // 2):
            self.generate_enemy()
        arcade.schedule(self.generate_enemy, ENEMY_GENERATE_INTERVAL)

    def on_draw(self) -> None:
        """Render the screen."""
        # Make sure variables needed are valid
        assert self.player is not None
        assert self.vector_field is not None

        # Clear the screen
        self.clear()

        # Activate our Camera
        self.game_camera.use()

        # Draw the various spritelists
        self.tile_sprites.draw(pixelated=True)
        self.bullet_sprites.draw(pixelated=True)
        self.enemy_sprites.draw(pixelated=True)
        self.player.draw(pixelated=True)
        self.enemy_indicator_bar_sprites.draw()

        # Draw stuff needed for the debug mode
        if DEBUG_GAME:
            # Draw the enemy debug circles
            for enemy in self.enemy_sprites:  # type: Enemy
                # Draw the enemy's view distance
                arcade.draw_circle_outline(
                    enemy.center_x,
                    enemy.center_y,
                    enemy.entity_data.view_distance * SPRITE_SIZE,
                    DEBUG_VIEW_DISTANCE,
                )

                # Draw the enemy's attack distance
                arcade.draw_circle_outline(
                    enemy.center_x,
                    enemy.center_y,
                    enemy.current_attack.attack_data.attack_range * SPRITE_SIZE,
                    DEBUG_ATTACK_DISTANCE,
                )

            # Check if the player's current attack is a melee attack or an area of
            # effect attack
            if isinstance(self.player.current_attack, MeleeAttack):
                # Calculate the two boundary points for the player fov
                half_angle = self.player.player_data.melee_degree // 2
                low_angle = math.radians(self.player.direction - half_angle)
                high_angle = math.radians(self.player.direction + half_angle)
                point_low = (
                    self.player.center_x
                    + math.cos(low_angle)
                    * SPRITE_SIZE
                    * self.player.current_attack.attack_data.attack_range,
                    self.player.center_y
                    + math.sin(low_angle)
                    * SPRITE_SIZE
                    * self.player.current_attack.attack_data.attack_range,
                )
                point_high = (
                    self.player.center_x
                    + math.cos(high_angle)
                    * SPRITE_SIZE
                    * self.player.current_attack.attack_data.attack_range,
                    self.player.center_y
                    + math.sin(high_angle)
                    * SPRITE_SIZE
                    * self.player.current_attack.attack_data.attack_range,
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
                # Draw the arc between the two making sure to double the width and
                # height since the radius is calculated, but we want the diameter
                arcade.draw_arc_outline(
                    self.player.center_x,
                    self.player.center_y,
                    math.dist(point_high, point_low) * 2,
                    self.player.current_attack.attack_data.attack_range
                    * SPRITE_SIZE
                    * 2,
                    DEBUG_ATTACK_DISTANCE,
                    math.degrees(low_angle),
                    math.degrees(high_angle),
                    2,
                )
            elif isinstance(self.player.current_attack, AreaOfEffectAttack):
                # Draw the player's attack range
                arcade.draw_circle_outline(
                    self.player.center_x,
                    self.player.center_y,
                    self.player.current_attack.attack_data.attack_range * SPRITE_SIZE,
                    DEBUG_ATTACK_DISTANCE,
                )

            # Draw the debug vector field lines
            lines: list[tuple[float, float]] = []
            for source, vector in self.vector_field.vector_dict.items():
                source_screen_x, source_screen_y = grid_pos_to_pixel(*source)
                lines.append((source_screen_x, source_screen_y))
                lines.append(
                    (
                        source_screen_x + vector[0] * SPRITE_SIZE,
                        source_screen_y + vector[1] * SPRITE_SIZE,
                    )
                )
            arcade.draw_lines(lines, DEBUG_VECTOR_FIELD_LINE)

            # Draw the enemy spawn locations
            points: list[tuple[float, float]] = []
            for location in self.possible_enemy_spawns:
                points.append(grid_pos_to_pixel(*location))
            arcade.draw_points(points, DEBUG_ENEMY_SPAWN_COLOR, DEBUG_ENEMY_SPAWN_SIZE)

        # Draw the gui on the screen
        self.gui_camera.use()
        self.player_gui_sprites.draw()
        self.player_status_text.value = f"Money: {str(self.player.money.value)}"
        self.player_status_text.draw()
        if self.nearest_item:
            self.item_text.text = self.nearest_item.item_text
            self.item_text.draw()

        # Draw the UI elements
        self.ui_manager.draw()

    def on_update(self, delta_time: float) -> None:
        """Process movement and game logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Make sure variables needed are valid
        assert self.physics_engine is not None
        assert self.player is not None

        # Check if the game should end
        # if self.player.health.value <= 0 or not self.enemy_sprites:
        #     arcade.exit()

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

        # Check for any nearby items
        item_collision = arcade.check_for_collision_with_list(
            self.player, self.item_sprites
        )
        if item_collision:
            # Set nearest_item since we are colliding with an item
            self.nearest_item = item_collision[0]
            logger.debug("Grabbed nearest item %r", self.nearest_item)
        else:
            # Reset nearest_item since we don't want to activate an item that the player
            # is not colliding with
            self.nearest_item = None

    def on_key_press(self, key: int, modifiers: int) -> None:
        """Process key press functionality.

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
        logger.debug("Received key press with key %r and modifiers %r", key, modifiers)
        match key:
            case arcade.key.W:
                self.player.up_pressed = True
            case arcade.key.S:
                self.player.down_pressed = True
            case arcade.key.A:
                self.player.left_pressed = True
            case arcade.key.D:
                self.player.right_pressed = True
            case arcade.key.E:
                if self.nearest_item:
                    with contextlib.suppress(AttributeError):
                        # Nearest item is a collectible. If this raises an error, then
                        # it is an item
                        self.nearest_item.item_pick_up()
            case arcade.key.R:
                if self.nearest_item:
                    self.nearest_item.item_use()
            case arcade.key.F:
                self.window.show_view(self.window.views["InventoryView"])
            case arcade.key.C:
                self.player.current_attack_index = min(
                    self.player.current_attack_index + 1,
                    len(self.player.attack_algorithms) - 1,
                )
            case arcade.key.Z:
                self.player.current_attack_index = max(
                    self.player.current_attack_index - 1, 0
                )
            case arcade.key.T:
                self.window.show_view(self.window.views["ShopView"])

    def on_key_release(self, key: int, modifiers: int) -> None:
        """Process key release functionality.

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

        # Find out what key was released
        logger.debug(
            "Received key release with key %r and modifiers %r", key, modifiers
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

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        """Process mouse button functionality.

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

        # Find out what mouse button was pressed
        logger.debug(
            "%r mouse button was pressed at position (%f, %f) with modifiers %r",
            button,
            x,
            y,
            modifiers,
        )
        if button is arcade.MOUSE_BUTTON_LEFT:
            # Make the player attack
            self.player.attack()

    def on_mouse_motion(self, x: float, y: float, *_) -> None:
        """Process mouse motion functionality.

        Parameters
        ----------
        x: float
            The x position of the mouse.
        y: float
            The y position of the mouse.
        """
        # Make sure variables needed are valid
        assert self.player is not None

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

    def generate_enemy(self, _: float = 1 / 60) -> None:
        """Generate an enemy outside the player's fov."""
        # Make sure variables needed are valid
        assert self.level_constants is not None
        assert self.physics_engine is not None

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
                    self.enemy_sprites
                ) or enemy.collides_with_sprite(self.player):
                    logger.debug(
                        "%r encountered a collision during generation. Retrying", enemy
                    )
                    tries -= 1
                    continue

                # Enemy position is good so add them to the spritelist and stop
                self.physics_engine.add_enemy(enemy)
                self.enemy_sprites.append(enemy)
                logger.debug("%d has been successfully generated")
                break

            # Enemy has failed to generate
            logger.debug("%d failed to be generated", enemy)

    def center_camera_on_player(self) -> None:
        """Centers the camera on the player."""
        # Make sure variables needed are valid
        assert self.level_constants is not None
        assert self.player is not None

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
        new_position = arcade.pymunk_physics_engine.Vec2(
            screen_center_x, screen_center_y
        )

        # Check if the camera position has changed
        if self.game_camera.position != new_position:
            # Move the camera to the new position
            self.game_camera.move_to(new_position)
