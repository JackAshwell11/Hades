"""Contains the functionality that manages the game and its events."""

from __future__ import annotations

# Builtin
import math
from typing import TYPE_CHECKING

# Custom
from hades.constructors import game_object_constructors
from hades.scenes.base import BaseScene
from hades.scenes.game.view import GameView
from hades.sprite import make_sprite
from hades_extensions.ecs import EventType, StatusEffectType
from hades_extensions.ecs.components import KinematicComponent, PythonSprite

if TYPE_CHECKING:
    from typing import ClassVar

__all__ = ("GameScene",)


class GameScene(BaseScene[GameView]):
    """Manages the game and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[GameView]] = GameView

    def add_callbacks(self: GameScene) -> None:
        """Set up the controller callbacks."""
        callbacks = [
            (EventType.GameObjectCreation, self.on_game_object_creation),
            (EventType.GameObjectDeath, self.on_game_object_death),
            (EventType.SpriteRemoval, self.on_sprite_removal),
            (EventType.StatusEffectUpdate, self.on_status_effect_update),
            (EventType.MoneyUpdate, self.on_money_update),
            (EventType.AttackCooldownUpdate, self.on_attack_cooldown_update),
            (EventType.RangedAttackSwitch, self.on_ranged_attack_switch),
        ]
        for event_type, callback in callbacks:
            self.model.registry.add_callback(  # type: ignore[call-overload]
                event_type,
                callback,
            )

    def on_show_view(self: GameScene) -> None:
        """Process show view functionality."""
        super().on_show_view()
        self.view.window.push_handlers(self.model.game_engine)

    def on_hide_view(self: GameScene) -> None:
        """Process hide view functionality."""
        super().on_hide_view()
        self.view.window.remove_handlers(self.model.game_engine)

    def on_update(self: GameScene, _: float) -> None:
        """Process game logic."""
        self.view.update(
            self.model.registry.get_component(
                self.model.player_id,
                KinematicComponent,
            ).get_position(),
        )

    def on_mouse_motion(self: GameScene, x: int, y: int, _: int, __: int) -> None:
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

    def on_game_object_creation(
        self: GameScene,
        game_object_id: int,
        position: tuple[float, float],
    ) -> None:
        """Process game object creation logic.

        Args:
            game_object_id: The ID of the newly created game object.
            position: The position of the newly created game object.
        """
        constructor = game_object_constructors[
            self.model.registry.get_game_object_type(game_object_id)
        ]
        sprite = make_sprite(self.model.registry, game_object_id, position, constructor)
        self.model.registry.get_component(game_object_id, PythonSprite).sprite = sprite
        self.view.add_sprite(sprite)

    def on_game_object_death(self: GameScene, game_object_id: int) -> None:
        """Process game object death logic.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        self.view.remove_progress_bars(game_object_id)
        self.on_sprite_removal(game_object_id)

    def on_sprite_removal(self: GameScene, game_object_id: int) -> None:
        """Process sprite removal logic.

        Args:
            game_object_id: The ID of the sprite to remove.
        """
        self.model.registry.get_component(
            game_object_id,
            PythonSprite,
        ).sprite.remove_from_sprite_lists()

    def on_money_update(self: GameScene, money: int) -> None:
        """Process money update logic.

        Args:
            money: The updated amount of money.
        """
        self.view.update_money_display(money)

    def on_attack_cooldown_update(
        self: GameScene,
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

    def on_ranged_attack_switch(self: GameScene, selected_attack: int) -> None:
        """Process ranged attack switch logic.

        Args:
            selected_attack: The selected attack type index.
        """
        self.view.update_ranged_attack_icon(selected_attack)

    def on_status_effect_update(
        self: GameScene,
        status_effects: dict[StatusEffectType, float],
    ) -> None:
        """Process status effect update logic.

        Args:
            status_effects: The status effects to display.
        """
        self.view.update_status_effects(status_effects)
