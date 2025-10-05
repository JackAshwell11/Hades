"""Contains the functionality that manages the game and its events."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.constructors import game_object_constructors
from hades.scenes.base import BaseScene
from hades.scenes.game.view import GameView
from hades.sprite import make_sprite
from hades_engine import EventType, add_callback

if TYPE_CHECKING:
    from typing import ClassVar

    from hades_engine.ecs import EffectType, GameObjectType

__all__ = ("GameScene",)


class GameScene(BaseScene[GameView]):
    """Manages the game and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[GameView]] = GameView

    def add_callbacks(self: GameScene) -> None:
        """Add callbacks for the scene."""
        callbacks = [
            (EventType.GameObjectCreation, self.on_game_object_creation),
            (EventType.GameObjectDeath, self.on_game_object_death),
            (EventType.PositionChanged, self.on_position_changed),
            (EventType.SpriteRemoval, self.on_sprite_removal),
            (EventType.StatusEffectUpdate, self.on_status_effect_update),
            (EventType.MoneyUpdate, self.on_money_update),
            (EventType.AttackCooldownUpdate, self.on_attack_cooldown_update),
            (EventType.RangedAttackSwitch, self.on_ranged_attack_switch),
            (EventType.GameOpen, self.on_game_open),
            (EventType.HealthChanged, self.on_health_changed),
            (EventType.ArmourChanged, self.on_armour_changed),
        ]
        for event_type, callback in callbacks:
            add_callback(event_type, callback)  # type: ignore[call-overload]

    def on_show_view(self: GameScene) -> None:
        """Process show view functionality."""
        super().on_show_view()
        self.view.window.push_handlers(self.model.game_engine)
        self.view.window.push_handlers(self.model.input_handler)

    def on_hide_view(self: GameScene) -> None:
        """Process hide view functionality."""
        super().on_hide_view()
        self.view.window.remove_handlers(self.model.game_engine)
        self.view.window.remove_handlers(self.model.input_handler)

    def on_update(self: GameScene, _: float) -> None:
        """Process game logic."""
        self.view.update(self.model.sprites[self.model.player_id].position)

    def on_game_object_creation(
        self: GameScene,
        game_object_id: int,
        game_object_type: GameObjectType,
    ) -> None:
        """Process game object creation logic.

        Args:
            game_object_id: The ID of the newly created game object.
            game_object_type: The type of the newly created game object.
        """
        constructor = game_object_constructors[game_object_type]
        sprite = make_sprite(game_object_id, constructor)
        self.model.sprites[game_object_id] = sprite
        self.view.add_sprite(sprite)

    def on_game_object_death(self: GameScene, game_object_id: int) -> None:
        """Process game object death logic.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        self.view.remove_progress_bars(game_object_id)
        self.on_sprite_removal(game_object_id)
        self.model.sprites.pop(game_object_id)

    def on_position_changed(
        self: GameScene,
        game_object_id: int,
        position: tuple[float, float],
    ) -> None:
        """Process position changed logic.

        Args:
            game_object_id: The ID of the game object to update.
            position: The new position of the game object.
        """
        self.model.sprites[game_object_id].position = position

    def on_sprite_removal(self: GameScene, game_object_id: int) -> None:
        """Process sprite removal logic.

        Args:
            game_object_id: The ID of the sprite to remove.
        """
        self.model.sprites[game_object_id].remove_from_sprite_lists()

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
    ) -> None:
        """Process attack cooldown update logic.

        Args:
            game_object_id: The ID of the game object to update.
            ranged_cooldown: The cooldown time for the ranged attack.
        """
        if game_object_id == self.model.player_id:
            self.view.update_attack_cooldown_display(ranged_cooldown)

    def on_ranged_attack_switch(self: GameScene, selected_attack: int) -> None:
        """Process ranged attack switch logic.

        Args:
            selected_attack: The selected attack type index.
        """
        self.view.update_ranged_attack_icon(selected_attack)

    def on_status_effect_update(
        self: GameScene,
        status_effects: dict[EffectType, float],
    ) -> None:
        """Process status effect update logic.

        Args:
            status_effects: The status effects to display.
        """
        self.view.update_status_effects(status_effects)

    def on_game_open(self: GameScene) -> None:
        """Process game open logic."""
        self.view.window.show_view(self)

    def on_health_changed(
        self: GameScene,
        game_object_id: int,
        percentage: float,
    ) -> None:
        """Process health changed logic.

        Args:
            game_object_id: The ID of the game object whose health has changed.
            percentage: The new health percentage.
        """
        self.view.update_progress_bar_value(
            self.model.sprites[game_object_id],
            percentage,
            health_bar=True,
        )

    def on_armour_changed(
        self: GameScene,
        game_object_id: int,
        percentage: float,
    ) -> None:
        """Process armour changed logic.

        Args:
            game_object_id: The ID of the game object whose armour has changed.
            percentage: The new armour percentage.
        """
        self.view.update_progress_bar_value(
            self.model.sprites[game_object_id],
            percentage,
            health_bar=False,
        )
