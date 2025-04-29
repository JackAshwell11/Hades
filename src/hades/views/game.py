"""Initialises and manages the main game."""

from __future__ import annotations

# Builtin
import logging
import math
from typing import Final, cast

# Pip
from arcade import SpriteList, color, key, schedule, unschedule
from arcade.camera.camera_2d import Camera2D
from arcade.gui import UIView
from pyglet import app

# Custom
from hades.constructors import game_object_constructors
from hades.progress_bar import ProgressBarGroup
from hades.sprite import AnimatedSprite, HadesSprite
from hades.views.game_ui import GameUI
from hades.views.player import PlayerView
from hades_extensions import GameEngine
from hades_extensions.ecs import EventType, GameObjectType, Registry
from hades_extensions.ecs.components import (
    Attack,
    KinematicComponent,
    Money,
    PythonSprite,
    StatusEffects,
)
from hades_extensions.ecs.systems import AttackSystem, InventorySystem, PhysicsSystem

__all__ = ("Game",)

# Constants
COLLECTIBLE_TYPES: Final[set[GameObjectType]] = {GameObjectType.HealthPotion}
ENEMY_GENERATE_INTERVAL: Final[int] = 1

# Get the logger
logger = logging.getLogger(__name__)


class Game(UIView):
    """Manages the game and its actions.

    Attributes:
        game_camera: The camera used for moving the viewport around the screen.
        sprites: The list of all sprites in the game.
        game_ui: The UI elements for the game.
        game_engine: The engine for the game which manages the game
        registry: The registry for the game which manages the game objects, components,
        and systems.
        nearest_item: The nearest item to the player.
        player: The ID of the player game object.
    """

    def __init__(self: Game) -> None:
        """Initialise the object."""
        super().__init__()
        self.game_camera: Camera2D = Camera2D()
        self.sprites: SpriteList[HadesSprite] = SpriteList[HadesSprite]()
        self.game_ui: GameUI = GameUI(self.ui)
        self.game_engine: GameEngine = cast("GameEngine", None)
        self.registry: Registry = cast("Registry", None)
        self.nearest_item: int = -1
        self.player: int = -1

        # Initialise the views
        self.window.views["Game"] = self
        self.window.views["InventoryView"] = PlayerView()

    def setup(self: Game, level: int, seed: int | None = None) -> None:
        """Set up the game.

        Args:
            level: The level to create a game for.
            seed: The seed to use for the game engine.
        """
        # Reset the game's state
        self.sprites.clear()
        if self.game_engine:
            unschedule(self.game_engine.generate_enemy)

        # Create the game engine and add the necessary callbacks
        self.game_engine = GameEngine(level, seed)
        self.registry = self.game_engine.get_registry()
        self.registry.add_callback(
            EventType.GameObjectCreation,
            self.on_game_object_creation,
        )
        self.registry.add_callback(EventType.GameObjectDeath, self.on_game_object_death)
        self.registry.add_callback(EventType.SpriteRemoval, self.on_sprite_removal)
        self.registry.add_callback(
            EventType.InventoryUpdate,
            self.window.views["InventoryView"].on_update_inventory,
        )

        # Set up the UI then finish setting up the rest of the game
        self.game_ui.setup()
        self.game_engine.create_game_objects()
        self.player = self.game_engine.player_id
        self.window.views["InventoryView"].setup(self.registry, self.player)
        schedule(self.game_engine.generate_enemy, ENEMY_GENERATE_INTERVAL)

        # Add the game engine's handlers to the window
        self.window.push_handlers(self.game_engine.on_key_press)
        self.window.push_handlers(self.game_engine.on_key_release)
        self.window.push_handlers(self.game_engine.on_mouse_press)

    def on_draw_before_ui(self: Game) -> None:
        """Render the screen before the UI elements are drawn."""
        self.window.background_color = color.BLACK
        self.game_camera.use()
        with self.window.ctx.enabled(self.window.ctx.DEPTH_TEST):
            self.sprites.draw(pixelated=True)

    def on_update(self: Game, _: float) -> None:
        """Process movement and game logic."""
        # Update the entities and the game UI elements
        self.sprites.update()
        self.nearest_item = self.registry.get_system(PhysicsSystem).get_nearest_item(
            self.player,
        )
        self.game_ui.update_progress_bars(self.game_camera)
        self.game_ui.update_info_box(self.registry, self.nearest_item)
        self.game_ui.update_money(self.registry.get_component(self.player, Money).money)
        self.game_ui.update_status_effects(
            self.registry.get_component(self.player, StatusEffects).active_effects,
        )

        # Check if the player has reached the goal
        if (
            self.nearest_item != -1
            and self.registry.get_component(
                self.nearest_item,
                PythonSprite,
            ).sprite.game_object_type
            == GameObjectType.Goal
        ):
            app.exit()

        # Position the camera on the player
        self.game_camera.position = self.registry.get_component(
            self.player,
            KinematicComponent,
        ).get_position()

    def on_fixed_update(self: Game, delta_time: float) -> None:
        """Process fixed update functionality.

        Args:
            delta_time: Time interval since the last time the function was called.
        """
        self.registry.update(delta_time)

    def on_key_release(self: Game, symbol: int, modifiers: int) -> None:
        """Process key release functionality.

        Args:
            symbol: The key that was hit.
            modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        logger.debug(
            "Received key release with key %r and modifiers %r",
            symbol,
            modifiers,
        )
        match symbol:
            case key.C:
                self.registry.get_system(InventorySystem).add_item_to_inventory(
                    self.player,
                    self.nearest_item,
                )
            case key.E:
                self.registry.get_system(InventorySystem).use_item(
                    self.player,
                    self.nearest_item,
                )
            case key.Z:
                self.registry.get_system(AttackSystem).previous_attack(self.player)
                self.game_ui.set_attack_algorithm(
                    self.registry.get_component(self.player, Attack).current_attack,
                )
            case key.X:
                self.registry.get_system(AttackSystem).next_attack(self.player)
                self.game_ui.set_attack_algorithm(
                    self.registry.get_component(self.player, Attack).current_attack,
                )
            case key.I:
                self.window.show_view(self.window.views["InventoryView"])

    def on_mouse_motion(self: Game, x: int, y: int, _: int, __: int) -> None:
        """Process mouse motion functionality.

        Args:
            x: The x position of the mouse.
            y: The y position of the mouse.
        """
        # Get the player's position
        kinematic_component = self.registry.get_component(
            self.player,
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

    def on_game_object_creation(self: Game, game_object_id: int) -> None:
        """Create a sprite from a newly created game object.

        Args:
            game_object_id: The ID of the newly created game object.
        """
        constructor = game_object_constructors[
            self.registry.get_game_object_type(game_object_id)
        ]
        sprite_class = (
            AnimatedSprite if len(constructor.texture_paths) > 1 else HadesSprite
        )
        sprite = sprite_class(
            self.game_engine.get_registry(),
            game_object_id,
            self.registry.get_component(
                game_object_id,
                KinematicComponent,
            ).get_position(),
            constructor,
        )
        self.sprites.append(sprite)
        if self.registry.has_component(game_object_id, PythonSprite):
            self.registry.get_component(game_object_id, PythonSprite).sprite = sprite

        # Create progress bars if needed
        if constructor.game_object_type == GameObjectType.Player:
            self.game_ui.player_ui.add(ProgressBarGroup(sprite))
        elif constructor.game_object_type == GameObjectType.Enemy:
            self.game_ui.progress_bar_groups.append(
                self.ui.add(ProgressBarGroup(sprite)),
            )

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

    def on_sprite_removal(self: Game, game_object_id: int) -> None:
        """Remove a sprite from the game.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        sprite = self.registry.get_component(game_object_id, PythonSprite).sprite
        if sprite.sprite_lists:
            sprite.remove_from_sprite_lists()

    def __repr__(self: Game) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Game (Current window={self.window})>"
