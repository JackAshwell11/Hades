"""Initialises and manages the main game."""

from __future__ import annotations

# Builtin
import logging
import math
import random
from typing import Final

# Pip
import arcade
from arcade.camera.camera_2d import Camera2D

# Custom
from hades.constructors import (
    GameObjectConstructor,
    game_object_constructors,
)
from hades.indicator_bar import INDICATOR_BAR_COMPONENTS, IndicatorBar
from hades.sprite import AnimatedSprite, Bullet, HadesSprite
from hades.views.player import PlayerView
from hades_extensions.game_objects import (
    SPRITE_SIZE,
    EventType,
    GameObjectType,
    Registry,
    Vec2d,
    grid_pos_to_pixel,
)
from hades_extensions.game_objects.components import (
    KeyboardMovement,
    KinematicComponent,
    PythonSprite,
    Stat,
    SteeringMovement,
)
from hades_extensions.game_objects.systems import AttackSystem, InventorySystem
from hades_extensions.generation import TileType, create_map

__all__ = ("Game",)

# Constants
COLLECTIBLE_TYPES: Final[set[GameObjectType]] = {GameObjectType.Potion}
ENEMY_GENERATE_INTERVAL: Final[int] = 1
ENEMY_GENERATION_DISTANCE: Final[int] = 5
ENEMY_RETRY_COUNT: Final[int] = 3
TOTAL_ENEMY_COUNT: Final[int] = 1

# Get the logger
logger = logging.getLogger(__name__)


# TODO: Improve the look of the indicator bars including their sizing, positioning, and
#  colouring.


class Game(arcade.View):
    """Manages the game and its actions.

    Attributes:
        game_camera: The camera used for moving the viewport around the screen.
        gui_camera: The camera used for visualising the GUI elements.
        tile_sprites: The sprite list for the tile game objects.
        entity_sprites: The sprite list for the entity game objects.
        item_sprites: The sprite list for the item game objects.
        nearest_item: The nearest item to the player.
        player_status_text: The text object used for displaying the player's health and
            armour.
        level_constants: Holds the constants for the current level.
        registry: The registry that manages the game objects, components, and systems.
        possible_enemy_spawns: A list of possible positions that enemies can spawn in.
        indicator_bars: A list of indicator bars that are currently being displayed.
        player: The player's sprite object.
    """

    def _create_sprite(
        self: Game,
        constructor: GameObjectConstructor,
        position: Vec2d,
    ) -> HadesSprite:
        """Create a sprite.

        Args:
            constructor: The constructor for the game object.
            position: The position of the game object in the grid.

        Returns:
            The created sprite object.
        """
        # Initialise the game object's constructor and a few other variables
        python_sprite = None
        game_object_id = -1
        sprite_class = (
            AnimatedSprite if len(constructor.texture_paths) > 1 else HadesSprite
        )

        # Create a game object if possible, adding a PythonSprite component if the game
        # object has other components
        if constructor.components:
            python_sprite = PythonSprite()
            game_object_id = self.registry.create_game_object(
                constructor.game_object_type,
                position,
                [*constructor.components, python_sprite],
            )

        # Create a sprite and add its ID to the dictionary
        sprite = sprite_class(self.registry, game_object_id, position, constructor)
        if python_sprite:
            python_sprite.sprite = sprite

        # Add all the indicator bars to the game
        indicator_bar_offset = 0
        for component in constructor.components:
            if type(component) in INDICATOR_BAR_COMPONENTS and isinstance(
                component,
                Stat,
            ):
                self.indicator_bars.append(
                    IndicatorBar(
                        sprite,
                        component,
                        self.indicator_bar_sprites,
                        indicator_bar_offset,
                    ),
                )
                indicator_bar_offset += 1
        return sprite

    def __init__(self: Game, level: int) -> None:
        """Initialise the object.

        Args:
            level: The level to create a game for.
        """
        super().__init__()
        # Arcade types
        self.game_camera: Camera2D = Camera2D()
        self.gui_camera: Camera2D = Camera2D()
        self.tile_sprites: arcade.SpriteList[HadesSprite] = arcade.SpriteList[
            HadesSprite
        ]()
        self.entity_sprites: arcade.SpriteList[HadesSprite] = arcade.SpriteList[
            HadesSprite
        ]()
        self.item_sprites: arcade.SpriteList[HadesSprite] = arcade.SpriteList[
            HadesSprite
        ]()
        self.indicator_bar_sprites: arcade.SpriteList[arcade.SpriteSolidColor] = (
            arcade.SpriteList[arcade.SpriteSolidColor]()
        )
        self.nearest_item: list[HadesSprite] = []
        self.player_status_text: arcade.Text = arcade.Text(
            "Money: 0",
            10,
            10,
            font_size=20,
        )

        # Custom types
        generation_result, self.level_constants = create_map(level)
        self.registry: Registry = Registry()

        # Custom collections
        self.possible_enemy_spawns: list[Vec2d] = []
        self.indicator_bars: list[IndicatorBar] = []

        # Initialise all the systems then the game objects
        self.registry.add_systems()
        for count, tile in enumerate(generation_result):
            # Skip all empty tiles
            if tile in {TileType.Empty, TileType.Obstacle}:
                continue

            # Get the screen position from the grid position
            position = Vec2d(
                count % self.level_constants.width,
                count // self.level_constants.width,
            )

            # Determine the type of the tile
            if tile == TileType.Wall:
                self.tile_sprites.append(
                    self._create_sprite(
                        game_object_constructors[GameObjectType.Wall](),
                        position,
                    ),
                )
            else:
                if tile == TileType.Player:
                    self.player = self._create_sprite(
                        game_object_constructors[GameObjectType.Player](),
                        position,
                    )
                    self.entity_sprites.append(self.player)
                elif tile == TileType.Potion:
                    self.item_sprites.append(
                        self._create_sprite(
                            game_object_constructors[GameObjectType.Potion](),
                            position,
                        ),
                    )

                # Make the game object's backdrop a floor
                self.tile_sprites.append(
                    self._create_sprite(
                        game_object_constructors[GameObjectType.Floor](),
                        position,
                    ),
                )
                self.possible_enemy_spawns.append(position)

        # Create the required views for the game
        inventory_view = PlayerView(
            self.registry,
            self.player.game_object_id,
            self.item_sprites,
        )
        self.window.views["InventoryView"] = inventory_view

        # Add the callbacks to the registry
        self.registry.add_callback(EventType.BulletCreation, self.on_bullet_creation)
        self.registry.add_callback(EventType.GameObjectDeath, self.on_game_object_death)
        self.registry.add_callback(
            EventType.InventoryUpdate,
            inventory_view.on_update_inventory,
        )

        # Generate half of the total enemies allowed to then schedule their generation
        for _ in range(TOTAL_ENEMY_COUNT // 2):
            self.generate_enemy()
        arcade.schedule(self.generate_enemy, ENEMY_GENERATE_INTERVAL)

    def on_draw(self: Game) -> None:
        """Render the screen."""
        # Clear the screen and set the background colour
        self.clear()
        self.window.background_color = arcade.color.BLACK

        # Activate our game camera
        self.game_camera.use()

        # Draw the various spritelists
        self.tile_sprites.draw(pixelated=True)
        self.item_sprites.draw(pixelated=True)
        self.entity_sprites.draw(pixelated=True)
        self.indicator_bar_sprites.draw()

        # Draw the gui on the screen
        self.gui_camera.use()
        self.player_status_text.draw()

    def on_update(self: Game, delta_time: float) -> None:
        """Process movement and game logic.

        Args:
            delta_time: Time interval since the last time the function was called.
        """
        # Update the systems and entities
        self.registry.update(delta_time)
        self.entity_sprites.update()

        # Find the nearest item to the player
        self.nearest_item = self.player.collides_with_list(self.item_sprites)

        # Update the indicator bars
        for indicator_bar in self.indicator_bars:
            indicator_bar.on_update(delta_time)

        # Position the camera on the player
        self.game_camera.position = self.player.position

    def on_key_press(self: Game, symbol: int, modifiers: int) -> None:
        """Process key press functionality.

        Args:
            symbol: The key that was hit.
            modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        player_movement = self.registry.get_component(
            self.player.game_object_id,
            KeyboardMovement,
        )
        logger.debug(
            "Received key press with key %r and modifiers %r",
            symbol,
            modifiers,
        )
        match symbol:
            case arcade.key.W:
                player_movement.moving_north = True
            case arcade.key.S:
                player_movement.moving_south = True
            case arcade.key.A:
                player_movement.moving_west = True
            case arcade.key.D:
                player_movement.moving_east = True

    def on_key_release(self: Game, symbol: int, modifiers: int) -> None:
        """Process key release functionality.

        Args:
            symbol: The key that was hit.
            modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        player_movement = self.registry.get_component(
            self.player.game_object_id,
            KeyboardMovement,
        )
        logger.debug(
            "Received key release with key %r and modifiers %r",
            symbol,
            modifiers,
        )
        match symbol:
            case arcade.key.W:
                player_movement.moving_north = False
            case arcade.key.S:
                player_movement.moving_south = False
            case arcade.key.A:
                player_movement.moving_west = False
            case arcade.key.D:
                player_movement.moving_east = False
            case arcade.key.C:
                if (
                    self.nearest_item
                    and self.nearest_item[0].game_object_type in COLLECTIBLE_TYPES
                    and self.registry.get_system(InventorySystem).add_item_to_inventory(
                        self.player.game_object_id,
                        self.nearest_item[0].game_object_id,
                    )
                ):
                    self.nearest_item[0].remove_from_sprite_lists()
            case arcade.key.E:
                if self.nearest_item and self.registry.get_system(
                    InventorySystem,
                ).use_item(
                    self.player.game_object_id,
                    self.nearest_item[0].game_object_id,
                ):
                    self.nearest_item[0].remove_from_sprite_lists()
            case arcade.key.Z:
                self.registry.get_system(AttackSystem).previous_attack(
                    self.player.game_object_id,
                )
            case arcade.key.X:
                self.registry.get_system(AttackSystem).next_attack(
                    self.player.game_object_id,
                )
            case arcade.key.I:
                self.window.show_view(self.window.views["InventoryView"])

    def on_mouse_press(self: Game, x: int, y: int, button: int, modifiers: int) -> None:
        """Process mouse button functionality.

        Args:
            x: The x position of the mouse.
            y: The y position of the mouse.
            button: Which button was hit.
            modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
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
            self.registry.get_system(AttackSystem).do_attack(
                self.player.game_object_id,
                [
                    game_object.game_object_id
                    for game_object in self.entity_sprites
                    if game_object.game_object_type == GameObjectType.Enemy
                ],
            )

    def on_mouse_motion(self: Game, x: int, y: int, _: int, __: int) -> None:
        """Process mouse motion functionality.

        Args:
            x: The x position of the mouse.
            y: The y position of the mouse.
        """
        # Get the player's position
        kinematic_component = self.registry.get_component(
            self.player.game_object_id,
            KinematicComponent,
        )
        player_x, player_y = kinematic_component.get_position()

        # Transform the mouse from window space to world space using the camera position
        # then set the player's rotation
        kinematic_component.set_rotation(
            math.atan2(
                y + self.game_camera.position[1] - self.window.height / 2 - player_y,
                x + self.game_camera.position[0] - self.window.width / 2 - player_x,
            ),
        )

    def on_bullet_creation(self: Game, game_object_id: int) -> None:
        """Create a bullet game object.

        Args:
            game_object_id: The ID of the created bullet game object.
        """
        bullet = Bullet(self.registry, game_object_id)
        self.registry.get_component(game_object_id, PythonSprite).sprite = bullet
        self.entity_sprites.append(bullet)

    def on_game_object_death(self: Game, game_object_id: int) -> None:
        """Remove a game object from the game.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        # Delete all the indicator bars for the game object
        indicator_bars_to_remove = []
        for indicator_bar in self.indicator_bars:
            if indicator_bar.target_sprite.game_object_id == game_object_id:
                indicator_bar.actual_bar.remove_from_sprite_lists()
                indicator_bar.background_box.remove_from_sprite_lists()
                indicator_bars_to_remove.append(indicator_bar)
        for indicator_bar in indicator_bars_to_remove:
            self.indicator_bars.remove(indicator_bar)

        # Remove the sprite from the game
        game_object = self.registry.get_component(game_object_id, PythonSprite).sprite
        game_object.remove_from_sprite_lists()
        if game_object.game_object_type == GameObjectType.Player:
            arcade.exit()

    def generate_enemy(self: Game, _: float = 1 / 60) -> None:
        """Generate an enemy outside the player's fov."""
        if (len(self.entity_sprites) - 1) >= TOTAL_ENEMY_COUNT:
            return

        # Enemy limit is not reached so try to initialise a new enemy game object
        # ENEMY_RETRY_COUNT times
        random.shuffle(self.possible_enemy_spawns)
        for position in self.possible_enemy_spawns[:ENEMY_RETRY_COUNT]:
            if (
                arcade.get_sprites_at_point(
                    grid_pos_to_pixel(position),
                    self.entity_sprites,
                )
                or math.dist(self.player.position, tuple(position))
                < ENEMY_GENERATION_DISTANCE * SPRITE_SIZE
            ):
                continue

            # Set the required data for the steering to correctly function
            new_sprite = self._create_sprite(
                game_object_constructors[GameObjectType.Enemy](),
                position,
            )
            self.entity_sprites.append(new_sprite)
            steering_movement = self.registry.get_component(
                new_sprite.game_object_id,
                SteeringMovement,
            )
            steering_movement.target_id = self.player.game_object_id
            return

    def __repr__(self: Game) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Game (Current window={self.window})>"
