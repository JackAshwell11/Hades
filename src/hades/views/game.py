"""Initialises and manages the main game."""

from __future__ import annotations

# Builtin
import logging
import math
import random
from typing import Final

# Pip
from arcade import (
    MOUSE_BUTTON_LEFT,
    SpriteList,
    color,
    get_sprites_at_point,
    key,
    schedule,
)
from arcade.camera.camera_2d import Camera2D
from arcade.gui import UIView
from pyglet import app

# Custom
from hades.constructors import GameObjectConstructor, game_object_constructors
from hades.progress_bar import ProgressBarGroup
from hades.sprite import AnimatedSprite, Bullet, HadesSprite
from hades.views.game_ui import GameUI
from hades.views.player import PlayerView
from hades_extensions.ecs import (
    SPRITE_SIZE,
    EventType,
    GameObjectType,
    Registry,
    Vec2d,
)
from hades_extensions.ecs.components import (
    Attack,
    KeyboardMovement,
    KinematicComponent,
    Money,
    PythonSprite,
    StatusEffect,
    SteeringMovement,
)
from hades_extensions.ecs.systems import AttackSystem, InventorySystem
from hades_extensions.generation import TileType, create_map

__all__ = ("Game",)

# Constants
COLLECTIBLE_TYPES: Final[set[GameObjectType]] = {GameObjectType.HealthPotion}
ENEMY_GENERATE_INTERVAL: Final[int] = 1
ENEMY_GENERATION_DISTANCE: Final[int] = 5
ENEMY_RETRY_COUNT: Final[int] = 3

# Get the logger
logger = logging.getLogger(__name__)


class Game(UIView):
    """Manages the game and its actions.

    Attributes:
        game_camera: The camera used for moving the viewport around the screen.
        tile_sprites: The sprite list for the tile game objects.
        entity_sprites: The sprite list for the entity game objects.
        item_sprites: The sprite list for the item game objects.
        nearest_item: The nearest item to the player.
        level_constants: Holds the constants for the current level.
        registry: The registry that manages the game objects, components, and systems.
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
        game_object_id = -1
        sprite_class = (
            AnimatedSprite if len(constructor.texture_paths) > 1 else HadesSprite
        )

        # Create a game object if possible, adding a PythonSprite component if the game
        # object has other components
        if constructor.components:
            game_object_id = self.registry.create_game_object(
                constructor.game_object_type,
                position,
                [*constructor.components],
            )

        # Create a sprite and add its ID to the dictionary
        sprite = sprite_class(self.registry, game_object_id, position, constructor)
        if self.registry.has_component(game_object_id, PythonSprite):
            self.registry.get_component(game_object_id, PythonSprite).sprite = sprite

        # Create progress bars if needed
        if constructor.game_object_type == GameObjectType.Player:
            self.game_ui.player_ui.add(ProgressBarGroup(sprite))
        elif constructor.game_object_type == GameObjectType.Enemy:
            progress_bar_group = self.ui.add(ProgressBarGroup(sprite))
            self.game_ui.progress_bar_groups.append(progress_bar_group)
        return sprite

    def __init__(self: Game, level: int) -> None:
        """Initialise the object.

        Args:
            level: The level to create a game for.
        """
        super().__init__()
        # Arcade types
        self.game_camera: Camera2D = Camera2D()
        self.tile_sprites: SpriteList[HadesSprite] = SpriteList[HadesSprite]()
        self.entity_sprites: SpriteList[HadesSprite] = SpriteList[HadesSprite]()
        self.item_sprites: SpriteList[HadesSprite] = SpriteList[HadesSprite]()
        self.nearest_item: list[HadesSprite] = []

        # Custom types
        generation_result, self.level_constants = create_map(level)
        self.registry: Registry = Registry()
        self.game_ui: GameUI = GameUI(self.ui)

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
                elif tile == TileType.HealthPotion:
                    self.item_sprites.append(
                        self._create_sprite(
                            game_object_constructors[GameObjectType.HealthPotion](),
                            position,
                        ),
                    )
                elif tile == TileType.Chest:
                    self.item_sprites.append(
                        self._create_sprite(
                            game_object_constructors[GameObjectType.Chest](),
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
        for _ in range(self.level_constants.enemy_limit // 2):
            self.generate_enemy()
        schedule(self.generate_enemy, ENEMY_GENERATE_INTERVAL)

    def on_draw_before_ui(self: Game) -> None:
        """Render the screen before the UI elements are drawn."""
        # Set the background colour and activate our game camera
        self.window.background_color = color.BLACK
        self.game_camera.use()

        # Draw the various spritelists
        self.tile_sprites.draw(pixelated=True)
        self.item_sprites.draw(pixelated=True)
        self.entity_sprites.draw(pixelated=True)

    def on_update(self: Game, delta_time: float) -> None:
        """Process movement and game logic.

        Args:
            delta_time: Time interval since the last time the function was called.
        """
        # Update the systems and entities
        self.registry.update(delta_time)
        self.entity_sprites.update()

        # Update the game UI elements
        self.nearest_item = self.player.collides_with_list(self.item_sprites)
        self.game_ui.update_progress_bars(self.game_camera)
        self.game_ui.update_info_box(self.nearest_item)
        self.game_ui.update_money(
            self.registry.get_component(self.player.game_object_id, Money).money,
        )
        self.game_ui.update_status_effects(
            self.registry.get_component(
                self.player.game_object_id,
                StatusEffect,
            ).applied_effects,
        )

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
            case key.W:
                player_movement.moving_north = True
            case key.S:
                player_movement.moving_south = True
            case key.A:
                player_movement.moving_west = True
            case key.D:
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
            case key.W:
                player_movement.moving_north = False
            case key.S:
                player_movement.moving_south = False
            case key.A:
                player_movement.moving_west = False
            case key.D:
                player_movement.moving_east = False
            case key.C:
                if (
                    self.nearest_item
                    and self.nearest_item[0].game_object_type in COLLECTIBLE_TYPES
                    and self.registry.get_system(InventorySystem).add_item_to_inventory(
                        self.player.game_object_id,
                        self.nearest_item[0].game_object_id,
                    )
                ):
                    self.nearest_item[0].remove_from_sprite_lists()
            case key.E:
                if self.nearest_item and self.registry.get_system(
                    InventorySystem,
                ).use_item(
                    self.player.game_object_id,
                    self.nearest_item[0].game_object_id,
                ):
                    self.nearest_item[0].remove_from_sprite_lists()
            case key.Z:
                self.registry.get_system(AttackSystem).previous_attack(
                    self.player.game_object_id,
                )
                self.game_ui.set_attack_algorithm(
                    self.registry.get_component(
                        self.player.game_object_id,
                        Attack,
                    ).current_attack,
                )
            case key.X:
                self.registry.get_system(AttackSystem).next_attack(
                    self.player.game_object_id,
                )
                self.game_ui.set_attack_algorithm(
                    self.registry.get_component(
                        self.player.game_object_id,
                        Attack,
                    ).current_attack,
                )
            case key.I:
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
        if button is MOUSE_BUTTON_LEFT:
            self.registry.get_system(AttackSystem).do_attack(
                (self.player.game_object_id, self.player.game_object_type),
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
        # Remove the sprite from the game
        self.game_ui.on_game_object_death(game_object_id)
        game_object = self.registry.get_component(game_object_id, PythonSprite).sprite
        game_object.remove_from_sprite_lists()
        if game_object.game_object_type == GameObjectType.Player:
            app.exit()

    def generate_enemy(self: Game, _: float = 1 / 60) -> None:
        """Generate an enemy outside the player's fov."""
        # Check if we have the maximum number of enemies
        if (len(self.entity_sprites) - 1) >= self.level_constants.enemy_limit:
            return

        # Enemy limit is not reached so try to initialise a new enemy game object
        # ENEMY_RETRY_COUNT times
        tile_positions = [
            floor.position
            for floor in self.tile_sprites
            if floor.game_object_type == GameObjectType.Floor
        ]
        random.shuffle(tile_positions)
        for position in tile_positions[:ENEMY_RETRY_COUNT]:
            if (
                get_sprites_at_point(position, self.entity_sprites)
                or math.dist(self.player.position, position)
                < ENEMY_GENERATION_DISTANCE * SPRITE_SIZE
            ):
                continue

            # Set the required data for the steering to correctly function
            new_sprite = self._create_sprite(
                game_object_constructors[GameObjectType.Enemy](),
                Vec2d(position[0] // SPRITE_SIZE, position[1] // SPRITE_SIZE),
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
