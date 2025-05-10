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
from hades.sprite import AnimatedSprite, HadesSprite
from hades.views.game_ui import GameUI
from hades.views.player import PlayerView
from hades_extensions import GameEngine
from hades_extensions.ecs import EventType, GameObjectType, Registry
from hades_extensions.ecs.components import KinematicComponent, PythonSprite

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
        self.player: int = -1

        # Initialise the views
        self.window.views["Game"] = self
        self.window.views["InventoryView"] = PlayerView()
        logger.debug("Initialised game view")

    def setup(self: Game, level: int, seed: int | None = None) -> None:
        """Set up the game.

        Args:
            level: The level to create a game for.
            seed: The seed to use for the game engine.
        """
        # Reset the game's state
        self.sprites.clear()
        logger.debug("Cleared sprites")
        if self.game_engine:
            unschedule(self.game_engine.generate_enemy)
            logger.debug("Unscheduling enemy generation")

        # Create the game engine and add the necessary callbacks
        self.game_engine = GameEngine(level, seed)
        logger.debug("Created game engine")
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
        self.registry.add_callback(
            EventType.StatusEffectUpdate,
            self.game_ui.on_status_effect_update,
        )
        self.registry.add_callback(
            EventType.MoneyUpdate,
            self.game_ui.on_money_update,
        )
        self.registry.add_callback(
            EventType.AttackCooldownUpdate,
            self.game_ui.on_attack_cooldown_update,
        )
        self.registry.add_callback(
            EventType.RangedAttackSwitch,
            self.game_ui.on_ranged_attack_switch,
        )

        # Set up the UI then finish setting up the rest of the game
        self.game_ui.setup()
        logger.debug("Set up game UI")
        self.game_engine.create_game_objects()
        self.game_ui.player_id = self.game_engine.player_id
        self.player = self.game_engine.player_id
        self.window.views["InventoryView"].setup(self.registry, self.player)
        logger.debug("Set up inventory view")
        schedule(self.game_engine.generate_enemy, ENEMY_GENERATE_INTERVAL)
        logger.debug("Scheduled enemy generation")

        # Add the game engine's handlers to the window
        self.window.push_handlers(self.game_engine.on_update)
        self.window.push_handlers(self.game_engine.on_fixed_update)
        self.window.push_handlers(self.game_engine.on_key_press)
        self.window.push_handlers(self.game_engine.on_key_release)
        self.window.push_handlers(self.game_engine.on_mouse_press)
        logger.debug("Added game engine handlers to window")

    def on_draw_before_ui(self: Game) -> None:
        """Render the screen before the UI elements are drawn."""
        self.window.background_color = color.BLACK
        self.game_camera.use()
        with self.window.ctx.enabled(self.window.ctx.DEPTH_TEST):
            self.sprites.draw(pixelated=True)

    def on_update(self: Game, _: float) -> None:
        """Process movement and game logic."""
        # Update the entities
        self.sprites.update()

        # Position the camera on the player
        self.game_camera.position = self.registry.get_component(
            self.player,
            KinematicComponent,
        ).get_position()

        # Update the UI elements
        self.game_ui.update_progress_bars(self.game_camera)

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
        logger.debug("Received game object creation event for %d", game_object_id)
        constructor = game_object_constructors[
            self.registry.get_game_object_type(game_object_id)
        ]
        sprite_class = (
            AnimatedSprite if len(constructor.texture_paths) > 1 else HadesSprite
        )
        logger.debug(
            "Initialising sprite class %s for %d",
            sprite_class,
            game_object_id,
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
        if constructor.progress_bars:
            self.game_ui.add_progress_bar(sprite)
            logger.debug("Created progress bar for %d", game_object_id)

    def on_game_object_death(self: Game, game_object_id: int) -> None:
        """Remove a game object from the game.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        logger.debug("Received game object death event for %d", game_object_id)
        self.game_ui.on_game_object_death(game_object_id)
        game_object = self.registry.get_component(game_object_id, PythonSprite).sprite
        game_object.remove_from_sprite_lists()
        if game_object.game_object_type == GameObjectType.Player:
            logger.info("Player has died, exiting game")
            app.exit()

    def on_sprite_removal(self: Game, game_object_id: int) -> None:
        """Remove a sprite from the game.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        logger.debug("Received sprite removal event for %d", game_object_id)
        sprite = self.registry.get_component(game_object_id, PythonSprite).sprite
        if sprite.sprite_lists:
            sprite.remove_from_sprite_lists()

    def __repr__(self: Game) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Game (Current window={self.window})>"
