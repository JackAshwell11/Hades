"""Manages the rendering of game elements on the screen."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final

# Pip
from arcade import SpriteList
from arcade.camera.camera_2d import Camera2D
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIImage, UILabel

# Custom
from hades import UI_PADDING
from hades.constructors import IconType
from hades.progress_bar import PROGRESS_BAR_HEIGHT, ProgressBar
from hades.scenes.base.view import BaseView
from hades.sprite import HadesSprite
from hades_extensions.ecs import SPRITE_SIZE, EffectType, GameObjectType

if TYPE_CHECKING:
    from hades.window import HadesWindow

__all__ = ("GameView", "StateIndicator")

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
            texture=icon_type.get_texture(),
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
        self.icon.texture = icon_type.get_texture()

    def update_value(self: StateIndicator, value: float) -> None:
        """Update the value of the label.

        Args:
            value: The value to set the label to.
        """
        self.label.text = str(round(value))


class GameView(BaseView):
    """Manages the rendering of game elements on the screen.

    Attributes:
        game_camera: The camera for the game.
        sprites: The sprites to render.
        progress_bars: The progress bars to display on the screen.
        left_layout: The layout for the left side of the screen.
        money_indicator: The money indicator.
        attack_type_layout: The layout for the attack type indicators.
        status_effect_layout: The layout for the status effect indicators.
        effect_indicators: The status effect indicators.
    """

    __slots__ = (
        "attack_type_layout",
        "effect_indicators",
        "game_camera",
        "left_layout",
        "money_indicator",
        "progress_bars",
        "sprites",
        "status_effect_layout",
    )

    def _setup_layout(self: GameView) -> None:
        """Set up the layout for the view."""
        self.left_layout.add(self.money_indicator)
        left_anchor = UIAnchorLayout()
        left_anchor.add(
            self.left_layout.with_padding(top=UI_PADDING * 2, left=UI_PADDING * 2),
            anchor_x="left",
            anchor_y="top",
        )
        self.ui.add(left_anchor)
        right_layout = UIBoxLayout(align="right", space_between=SPRITE_SIZE / 2)
        right_layout.add(self.attack_type_layout)
        right_layout.add(self.status_effect_layout)
        right_anchor = UIAnchorLayout()
        right_anchor.add(
            right_layout.with_padding(top=UI_PADDING * 2, right=UI_PADDING * 2),
            anchor_x="right",
            anchor_y="top",
        )
        self.ui.add(right_anchor)

    def __init__(self: GameView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the game.
        """
        self.game_camera: Camera2D = Camera2D()
        self.sprites: SpriteList[HadesSprite] = SpriteList[HadesSprite]()
        self.progress_bars: list[ProgressBar] = []
        self.left_layout: UIBoxLayout = UIBoxLayout(align="left")
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
        self.effect_indicators: dict[EffectType, StateIndicator] = {
            EffectType.Regeneration: StateIndicator(
                IconType.REGENERATION,
                reverse=True,
            ),
            EffectType.Poison: StateIndicator(IconType.POISON, reverse=True),
        }
        super().__init__(window)

    def draw(self: GameView) -> None:
        """Draw the game elements."""
        self.window.clear()
        self.game_camera.use()
        with self.window.ctx.enabled(self.window.ctx.DEPTH_TEST):
            self.sprites.draw(pixelated=True)
        self.ui.draw()

    def update(self: GameView, player_position: tuple[float, float]) -> None:
        """Update the game elements.

        Args:
            player_position: The position of the player.
        """
        self.sprites.update()
        self.game_camera.position = player_position
        self.update_progress_bars(self.game_camera)

    def add_sprite(self: GameView, sprite: HadesSprite) -> None:
        """Add a sprite to the game.

        Args:
            sprite: The sprite to add.
        """
        self.sprites.append(sprite)
        if sprite.constructor.progress_bars:
            self.add_progress_bar(sprite)

    def add_progress_bar(self: GameView, sprite: HadesSprite) -> None:
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
                self.left_layout.add(
                    progress_bar.with_padding(top=UI_PADDING, left=UI_PADDING),
                    index=0,
                )
            else:
                self.ui.add(progress_bar)

    def remove_progress_bars(self: GameView, game_object_id: int = -1) -> None:
        """Remove progress bars for a game object.

        Args:
            game_object_id: ID of the game object to remove progress bars for.
        """
        for progress_bar in self.progress_bars[:]:
            if progress_bar.target_sprite.game_object_id == game_object_id:
                self.ui.remove(progress_bar)
                if progress_bar in self.left_layout.children:
                    self.left_layout.remove(progress_bar)
                self.progress_bars.remove(progress_bar)

    def update_progress_bars(self: GameView, camera: Camera2D) -> None:
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

    def update_money_display(self: GameView, money: int) -> None:
        """Update the money indicator.

        Args:
            money: The updated amount of money.
        """
        self.money_indicator.update_value(money)

    def update_attack_cooldown_display(
        self: GameView,
        ranged_cooldown: float,
        melee_cooldown: float,
        special_cooldown: float,
    ) -> None:
        """Update the attack cooldown indicators.

        Args:
            ranged_cooldown: The cooldown time for the ranged attack.
            melee_cooldown: The cooldown time for the melee attack.
            special_cooldown: The cooldown time for the special attack.
        """
        self.attack_type_layout.children[0].update_value(ranged_cooldown)
        self.attack_type_layout.children[1].update_value(melee_cooldown)
        self.attack_type_layout.children[2].update_value(special_cooldown)

    def update_ranged_attack_icon(self: GameView, selected_attack: int) -> None:
        """Update the ranged attack type indicator.

        Args:
            selected_attack: The selected attack type index
        """
        self.attack_type_layout.children[0].update_icon(
            RANGED_ATTACK_ICON_MAP[selected_attack],
        )

    def update_status_effects(
        self: GameView,
        status_effects: dict[EffectType, float],
    ) -> None:
        """Update the status effects for the player.

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
