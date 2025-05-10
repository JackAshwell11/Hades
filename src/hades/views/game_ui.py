"""Initialises and manages the main game."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final

# Pip
from arcade.gui import (
    UIAnchorLayout,
    UIBoxLayout,
    UIImage,
    UILabel,
    UIManager,
)

# Custom
from hades import UI_PADDING
from hades.constructors import IconType
from hades.progress_bar import (
    PROGRESS_BAR_HEIGHT,
    ProgressBar,
)
from hades_extensions.ecs import SPRITE_SIZE, GameObjectType, StatusEffectType

if TYPE_CHECKING:
    from arcade.camera import Camera2D

    from hades.sprite import HadesSprite


# The icon types for the different ranged attack types
RANGED_ATTACK_ICON_MAP: Final[list[IconType]] = [
    IconType.SINGLE_BULLET,
    IconType.MULTI_BULLET,
]


class StateIndicator(UIBoxLayout):
    """Represents a state indicator with an icon and a label.

    Attributes:
        icon: The icon widget.
        label: The text label.
    """

    __slots__ = ("icon", "label")

    def __init__(
        self: StateIndicator,
        icon_type: IconType,
        *,
        reverse: bool = False,
    ) -> None:
        """Initialise the object.

        Args:
            icon_type: The icon type to use.
            reverse: Whether to reverse the icon and label order or not.
        """
        super().__init__(vertical=False)
        self.icon: UIImage = UIImage(
            texture=icon_type.value,
            width=SPRITE_SIZE,
            height=SPRITE_SIZE,
        )
        self.label: UILabel = UILabel("0", font_size=22, bold=True)

        # Add components in the requested order
        if reverse:
            self.add(self.label)
            self.add(self.icon)
        else:
            self.add(self.icon)
            self.add(self.label)

    def update_icon(self: StateIndicator, icon_type: IconType) -> None:
        """Update the icon of the label.

        Args:
            icon_type: The icon type to set the label to.
        """
        self.icon.texture = icon_type.value

    def update_value(self: StateIndicator, value: float) -> None:
        """Update the value of the label.

        Args:
            value: The value to set the label to.
        """
        self.label.text = str(round(value))


class GameUI:
    """Manages and updates the game UI.

    Attributes:
        ui: The UI manager to use.
        progress_bars: The progress bars to display on the screen.
        player_ui: The UI for the player.
        money_indicator: The money indicator.
        attack_type_layout: The layout for the attack types.
        status_effect_layout: The layout for the status effects.
        effect_indicators: The status effect indicators.
        player_id: The ID of the player.
    """

    __slots__ = (
        "attack_type_layout",
        "effect_indicators",
        "money_indicator",
        "player_id",
        "player_ui",
        "progress_bars",
        "status_effect_layout",
        "ui",
    )

    def __init__(self: GameUI, ui: UIManager) -> None:
        """Initialise the object.

        Args:
            ui: The UI manager to use.
        """
        self.ui: UIManager = ui
        self.progress_bars: list[ProgressBar] = []
        self.player_ui: UIBoxLayout = UIBoxLayout(align="left")
        self.money_indicator: StateIndicator = StateIndicator(IconType.MONEY)
        self.attack_type_layout: UIBoxLayout = UIBoxLayout(
            vertical=False,
            align="right",
            children=[
                StateIndicator(IconType.SINGLE_BULLET, reverse=True),
                StateIndicator(IconType.MELEE, reverse=True),
                StateIndicator(IconType.SPECIAL, reverse=True),
            ],
            space_between=SPRITE_SIZE / 2,
        )
        self.status_effect_layout: UIBoxLayout = UIBoxLayout(align="right")
        self.effect_indicators: dict[StatusEffectType, StateIndicator] = {
            StatusEffectType.Regeneration: StateIndicator(
                IconType.REGENERATION,
                reverse=True,
            ),
            StatusEffectType.Poison: StateIndicator(IconType.POISON, reverse=True),
        }
        self.player_id: int = -1

    def setup(self: GameUI) -> None:
        """Set up the game UI."""
        # Reset the UI's state
        self.progress_bars.clear()
        self.player_ui.clear()
        self.ui.clear()

        # Add the player UI elements
        self.player_ui.add(self.money_indicator)
        left_anchor = UIAnchorLayout(align="left")
        left_anchor.add(self.player_ui, anchor_x="left", anchor_y="top")
        self.ui.add(left_anchor)
        right_layout = UIBoxLayout(align="right", space_between=SPRITE_SIZE / 2)
        right_layout.add(
            self.attack_type_layout.with_padding(
                top=UI_PADDING * 2,
                left=UI_PADDING * 2,
            ),
        )
        right_layout.add(
            self.status_effect_layout.with_padding(
                top=UI_PADDING * 2,
                left=UI_PADDING * 2,
            ),
        )
        right_anchor = UIAnchorLayout()
        right_anchor.add(right_layout, anchor_x="right", anchor_y="top")
        self.ui.add(right_anchor)

    def add_progress_bar(self: GameUI, sprite: HadesSprite) -> None:
        """Add progress bars for a game object.

        Args:
            sprite: The sprite to create progress bars for.
        """
        for order, bar_data in enumerate(sprite.constructor.progress_bars):
            progress_bar = ProgressBar(
                (sprite, bar_data[0]),
                bar_data[1],
                bar_data[2],
                order,
            )
            self.progress_bars.append(progress_bar)
            if sprite.game_object_type == GameObjectType.Player:
                self.player_ui.add(
                    progress_bar.with_padding(top=UI_PADDING, left=UI_PADDING),
                    index=0,
                )
            else:
                self.ui.add(progress_bar)

    def update_progress_bars(self: GameUI, camera: Camera2D) -> None:
        """Update the progress bars on the screen.

        Args:
            camera: The camera to project the progress bars onto.
        """
        for progress_bar in self.progress_bars:
            screen_pos = camera.project(progress_bar.target_sprite.position)
            progress_bar.rect = progress_bar.rect.align_center_x(
                screen_pos.x,
            ).align_bottom(
                screen_pos.y
                + SPRITE_SIZE // 2
                + progress_bar.order * PROGRESS_BAR_HEIGHT,
            )

    def on_money_update(self: GameUI, money: int) -> None:
        """Update the money indicator on the screen.

        Args:
            money: The amount of money to display.
        """
        self.money_indicator.update_value(money)

    def on_attack_cooldown_update(
        self: GameUI,
        game_object_id: int,
        ranged_cooldown: float,
        melee_cooldown: float,
        special_cooldown: float,
    ) -> None:
        """Update the attack cooldown indicators on the screen.

        Args:
            game_object_id: The ID of the game object to update.
            ranged_cooldown: The cooldown time for the ranged attack.
            melee_cooldown: The cooldown time for the melee attack.
            special_cooldown: The cooldown time for the special attack.
        """
        if game_object_id == self.player_id:
            self.attack_type_layout.children[0].update_value(ranged_cooldown)
            self.attack_type_layout.children[1].update_value(melee_cooldown)
            self.attack_type_layout.children[2].update_value(special_cooldown)

    def on_ranged_attack_switch(self: GameUI, selected_attack: int) -> None:
        """Update the ranged attack type indicator on the screen.

        Args:
            selected_attack: The selected attack type.
        """
        self.attack_type_layout.children[0].update_icon(
            RANGED_ATTACK_ICON_MAP[selected_attack],
        )

    def on_status_effect_update(
        self: GameUI,
        status_effects: dict[StatusEffectType, float],
    ) -> None:
        """Update the status effects for a game object.

        Args:
            status_effects: The status effects to display.
        """
        for effect_type, indicator in self.effect_indicators.items():
            if effect_type in status_effects:
                if indicator not in self.status_effect_layout.children:
                    self.status_effect_layout.add(indicator)
                indicator.update_value(status_effects[effect_type])
            elif indicator in self.status_effect_layout.children:
                self.status_effect_layout.remove(indicator)

    def on_game_object_death(self: GameUI, game_object_id: int) -> None:
        """Remove a game object from the game.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        for group in self.progress_bars[:]:
            if group.target_sprite.game_object_id == game_object_id:
                self.ui.remove(group)
                self.progress_bars.remove(group)
