"""Manages the game flow and registry callbacks."""

from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING, Final

# Pip
from arcade import color, key, schedule, unschedule
from pyglet import app

# Custom
from hades import ViewType
from hades.constructors import game_object_constructors
from hades.sprite import make_sprite
from hades_extensions.ecs import (
    EventType,
    GameObjectType,
    StatusEffectType,
)
from hades_extensions.ecs.components import KinematicComponent, PythonSprite

if TYPE_CHECKING:
    from hades.game.model import GameModel
    from hades.game.view import GameView

__all__ = ("GameController",)

# The time interval between enemy generation attempts
ENEMY_GENERATE_INTERVAL: Final[int] = 1

# Get the logger for this module
logger = logging.getLogger(__name__)


class GameController:
    """Manages the game flow and registry callbacks.

    Attributes:
        model: The game model.
        view: The game renderer.
    """

    __slots__ = ("model", "view")

    def __init__(
        self: GameController,
        model: GameModel,
        view: GameView,
    ) -> None:
        """Initialise the object.

        Args:
            model: The model managing the game state.
            view: The renderer for the game.
        """
        self.model: GameModel = model
        self.view: GameView = view

    def setup(self: GameController) -> None:
        """Set up the controller."""
        callbacks = [
            (EventType.GameObjectCreation, self.on_game_object_creation),
            (EventType.GameObjectDeath, self.on_game_object_death),
            (
                EventType.InventoryUpdate,
                self.view.window.views[ViewType.PLAYER].on_update_inventory,
            ),
            (EventType.SpriteRemoval, self.on_sprite_removal),
            (EventType.StatusEffectUpdate, self.on_status_effect_update),
            (EventType.MoneyUpdate, self.on_money_update),
            (EventType.AttackCooldownUpdate, self.on_attack_cooldown_update),
            (EventType.RangedAttackSwitch, self.on_ranged_attack_switch),
        ]
        for event_type, callback in callbacks:
            self.model.registry.add_callback(event_type, callback)
        self.model.game_engine.create_game_objects()
        self.model.player_id = self.model.game_engine.player_id

    def show_view(self: GameController) -> None:
        """Process show view functionality."""
        self.view.window.background_color = color.BLACK
        self.view.ui.enable()
        self.view.window.push_handlers(self.model.game_engine)
        schedule(self.model.game_engine.generate_enemy, ENEMY_GENERATE_INTERVAL)

    def hide_view(self: GameController) -> None:
        """Process hide view functionality."""
        self.view.ui.disable()
        self.view.window.remove_handlers(self.model.game_engine)
        unschedule(self.model.game_engine.generate_enemy)

    def key_release(self: GameController, symbol: int, modifiers: int) -> None:
        """Process key release functionality.

        Args:
            symbol: The key that was hit.
            modifiers: The bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        logger.debug(
            "Received key release with key %r and modifiers %r",
            symbol,
            modifiers,
        )
        match symbol:
            case key.I:
                self.view.window.show_view(self.view.window.views[ViewType.PLAYER])

    def mouse_motion(self: GameController, x: int, y: int, _: int, __: int) -> None:
        """Process mouse motion functionality.

        Args:
            x: The x position of the mouse.
            y: The y position of the mouse.
        """
        kinematic_component = self.model.registry.get_component(
            self.model.player_id,
            KinematicComponent,
        )
        player_x, player_y = kinematic_component.get_position()
        kinematic_component.set_rotation(
            math.atan2(
                y
                + self.view.game_camera.position[1]
                - self.view.window.height / 2
                - player_y,
                x
                + self.view.game_camera.position[0]
                - self.view.window.width / 2
                - player_x,
            ),
        )

    def update(self: GameController, _: float) -> None:
        """Process game logic."""
        self.view.update(
            self.model.registry.get_component(
                self.model.player_id,
                KinematicComponent,
            ).get_position(),
        )

    def on_game_object_creation(self: GameController, game_object_id: int) -> None:
        """Process game object creation logic.

        Args:
            game_object_id: The ID of the newly created game object.
        """
        logger.debug("Received game object creation event for %d", game_object_id)
        constructor = game_object_constructors[
            self.model.registry.get_game_object_type(game_object_id)
        ]
        sprite = make_sprite(self.model.registry, game_object_id, constructor)
        self.view.add_sprite(sprite)
        if constructor.progress_bars:
            self.view.add_progress_bar(sprite)

    def on_game_object_death(self: GameController, game_object_id: int) -> None:
        """Process game object death logic.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        logger.debug("Received game object death event for %d", game_object_id)
        self.view.remove_progress_bars(game_object_id)
        game_object = self.model.registry.get_component(
            game_object_id,
            PythonSprite,
        ).sprite
        game_object.remove_from_sprite_lists()
        if (
            self.model.registry.get_game_object_type(game_object_id)
            == GameObjectType.Player
        ):
            logger.info("Player has died, exiting game")
            app.exit()

    def on_sprite_removal(self: GameController, game_object_id: int) -> None:
        """Process sprite removal logic.

        Args:
            game_object_id: The ID of the sprite to remove.
        """
        logger.debug("Received sprite removal event for %d", game_object_id)
        sprite = self.model.registry.get_component(game_object_id, PythonSprite).sprite
        if sprite.sprite_lists:
            sprite.remove_from_sprite_lists()

    def on_money_update(self: GameController, money: int) -> None:
        """Process money update logic.

        Args:
            money: The updated amount of money.
        """
        self.view.update_money_display(money)

    def on_attack_cooldown_update(
        self: GameController,
        game_object_id: int,
        ranged_cooldown: float,
        melee_cooldown: float,
        special_cooldown: float,
    ) -> None:
        """Process attack cooldown update logic.

        Args:
            game_object_id: The ID of the game object to update.
            ranged_cooldown: The cooldown time for the ranged attack.
            melee_cooldown: The cooldown time for the melee attack.
            special_cooldown: The cooldown time for the special attack.
        """
        if game_object_id == self.model.player_id:
            self.view.update_attack_cooldown_display(
                ranged_cooldown,
                melee_cooldown,
                special_cooldown,
            )

    def on_ranged_attack_switch(self: GameController, selected_attack: int) -> None:
        """Process ranged attack switch logic.

        Args:
            selected_attack: The selected attack type index.
        """
        self.view.update_ranged_attack_icon(selected_attack)

    def on_status_effect_update(
        self: GameController,
        status_effects: dict[StatusEffectType, float],
    ) -> None:
        """Process status effect update logic.

        Args:
            status_effects: The status effects to display.
        """
        self.view.update_status_effects(status_effects)
