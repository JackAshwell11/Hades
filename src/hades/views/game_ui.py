"""Initialises and manages the main game."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import color
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIGridLayout, UILabel, UIManager

# Custom
from hades.progress_bar import PROGRESS_BAR_DISTANCE, ProgressBarGroup
from hades.views import UI_BACKGROUND_COLOUR
from hades_extensions.ecs import AttackAlgorithm, GameObjectType
from hades_extensions.ecs.components import PythonSprite

if TYPE_CHECKING:
    from arcade.camera import Camera2D

    from hades_extensions.ecs import Registry, StatusEffectType
    from hades_extensions.ecs.components import Effect


class GameUI:
    """Manages and updates the game UI.

    Attributes:
        attack_algorithm_label: The label for the player's attack algorithm.
        info_box: The information box for the nearest item.
        progress_bar_groups: The progress bar groups to display on the screen.
        player_ui: The UI for the player.
        status_effect_layout: The layout for the status effects.
    """

    __slots__ = (
        "attack_algorithm_label",
        "info_box",
        "player_ui",
        "progress_bar_groups",
        "status_effect_layout",
        "ui",
    )

    def __init__(self: GameUI, ui: UIManager) -> None:
        """Initialise the object.

        Args:
            ui: The UI manager to use.
        """
        self.ui: UIManager = ui
        self.info_box: UILabel = UILabel(
            "",
            x=ui.window.width // 2,
            y=30,
            text_color=color.BLACK,
        ).with_background(color=UI_BACKGROUND_COLOUR)
        self.progress_bar_groups: list[ProgressBarGroup] = []

        # Initialise the player UI
        anchor = UIAnchorLayout()
        self.player_ui: UIGridLayout = anchor.add(
            UIGridLayout(row_count=4),
            anchor_x="left",
            anchor_y="top",
        )
        ui.add(anchor)

        # Add the money indicator to the player UI
        money_anchor = UIAnchorLayout()
        money_anchor.add(
            UILabel("Money: 0", text_color=color.BLACK).with_background(
                color=UI_BACKGROUND_COLOUR,
            ),
            anchor_x="left",
            anchor_y="top",
        )
        self.player_ui.add(money_anchor, row_num=1)

        # Add the status effect indicator to the player UI
        self.status_effect_layout: UIBoxLayout = UIBoxLayout(vertical=False)
        self.player_ui.add(self.status_effect_layout, row_num=2)

        # Add the player attack algorithm indicator to the player UI
        self.attack_algorithm_label: UILabel = UILabel(
            f"Attack Algorithm: {AttackAlgorithm.Ranged.name}",
            text_color=color.BLACK,
        ).with_background(color=UI_BACKGROUND_COLOUR)
        self.player_ui.add(self.attack_algorithm_label, row_num=3)

    def update_progress_bars(self: GameUI, camera: Camera2D) -> None:
        """Update the progress bars on the screen.

        Args:
            camera: The camera to project the progress bars onto.
        """
        for progress_bar_group in self.progress_bar_groups:
            if progress_bar_group.sprite.game_object_type == GameObjectType.Enemy:
                screen_pos = camera.project(
                    progress_bar_group.sprite.position,
                )
                ui_x, ui_y, _ = self.ui.camera.unproject(screen_pos)
                progress_bar_group.rect = progress_bar_group.rect.align_center_x(
                    ui_x,
                ).align_bottom(ui_y + PROGRESS_BAR_DISTANCE)

    def update_info_box(self: GameUI, registry: Registry, nearest_item: int) -> None:
        """Update the info box on the screen.

        Args:
            registry: The registry that manages the game objects, components, and
            systems.
            nearest_item: The nearest item to the player.
        """
        if nearest_item != -1 and not self.info_box.visible:
            sprite = registry.get_component(nearest_item, PythonSprite).sprite
            self.info_box.text = f"{sprite.game_object_type.name}: Collect (C), Use (E)"
            self.info_box.fit_content()
            self.info_box.rect = self.info_box.rect.align_x(self.ui.window.width // 2)
            self.info_box.visible = True
            self.ui.add(self.info_box)
        elif nearest_item == -1 and self.info_box.visible:
            self.info_box.visible = False
            self.ui.remove(self.info_box)

    def update_money(self: GameUI, money: int) -> None:
        """Update the money indicator on the screen.

        Args:
            money: The amount of money the player has.
        """
        self.player_ui.children[0].children[0].text = f"Money: {money}"

    def update_status_effects(
        self: GameUI,
        status_effects: dict[StatusEffectType, Effect],
    ) -> None:
        """Update the status effects indicator on the screen.

        Args:
            status_effects: The status effects to display.
        """
        if len(status_effects) != len(self.status_effect_layout.children):
            self.status_effect_layout.clear()

            # For each status effect, create a vertical box layout with two labels
            for status_effect_type, effect in status_effects.items():
                layout = UIBoxLayout()
                layout.add(
                    UILabel(
                        status_effect_type.name,
                        text_color=color.BLACK,
                    ).with_background(color=UI_BACKGROUND_COLOUR),
                )
                layout.add(
                    UILabel(
                        f"{effect.duration:.1f}",
                        text_color=color.BLACK,
                    ).with_background(color=UI_BACKGROUND_COLOUR),
                )
                self.status_effect_layout.add(layout)

    def set_attack_algorithm(self: GameUI, attack_algorithm: AttackAlgorithm) -> None:
        """Set the player's attack algorithm.

        Args:
            attack_algorithm: The player's attack algorithm.
        """
        self.attack_algorithm_label.text = f"Attack Algorithm: {attack_algorithm.name}"

    def on_game_object_death(self: GameUI, game_object_id: int) -> None:
        """Remove a game object from the game.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        # Delete all the progress bars for the game object
        to_remove = [
            group
            for group in self.progress_bar_groups
            if group.sprite.game_object_id == game_object_id
        ]
        for group in to_remove:
            self.ui.remove(group)
            self.progress_bar_groups.remove(group)
