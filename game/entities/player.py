from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from game.constants.entity import (
    ARMOUR_INDICATOR_BAR_COLOR,
    HEALTH_INDICATOR_BAR_COLOR,
    INDICATOR_BAR_BORDER_SIZE,
    MOVEMENT_FORCE,
    SPRITE_SIZE,
    AttackAlgorithmType,
    EntityID,
    UpgradeSection,
)
from game.constants.general import INVENTORY_HEIGHT, INVENTORY_WIDTH
from game.entities.base import Entity, IndicatorBar
from game.entities.upgrades import UpgradableSection
from game.melee_shader import MeleeShader

if TYPE_CHECKING:
    from game.constants.entity import BaseData
    from game.entities.base import CollectibleTile
    from game.views.game_view import Game

# Get the logger
logger = logging.getLogger(__name__)


class Player(Entity):
    """
    Represents the player character in the game.

    Parameters
    ----------
    game: Game
        The game view. This is passed so the player can have a reference to it.
    x: int
        The x position of the player in the game map.
    y: int
        The y position of the player in the game map.
    player_type: BaseData
        The raw data for this player.

    Attributes
    ----------
    melee_shader: MeleeShader
        The OpenGL shader used to find and attack any enemies within a specific distance
        around the player based on their direction.
    upgrade_sections: dict[UpgradeSection, UpgradableSection]
        A mapping of an upgrade section enum to an upgradable section object.
    inventory: list[game.entities.tile.Item]
        The list which stores the player's inventory.
    inventory_capacity: int
        The total capacity of the inventory.
    in_combat: bool
        Whether the player is in combat or not.
    left_pressed: bool
        Whether the left key is pressed or not.
    right_pressed: bool
        Whether the right key is pressed or not.
    up_pressed: bool
        Whether the up key is pressed or not.
    down_pressed: bool
        Whether the down key is pressed or not.
    """

    # Class variables
    entity_id: EntityID = EntityID.PLAYER

    def __init__(self, game: Game, x: int, y: int, player_type: BaseData) -> None:
        super().__init__(game, x, y, player_type)
        self.melee_shader: MeleeShader = MeleeShader(self.game)
        self.upgrade_sections: dict[UpgradeSection, UpgradableSection] = {
            upgrade_data.section_type: UpgradableSection(self, upgrade_data, 1)
            for upgrade_data in self.upgrade_data
        }
        self.health_bar: IndicatorBar = IndicatorBar(
            self,
            self.game.player_gui_sprites,
            (0, 0),
            HEALTH_INDICATOR_BAR_COLOR,
            border_size=INDICATOR_BAR_BORDER_SIZE,
            scale=4,
        )
        self.health_bar.position = (
            self.health_bar.half_bar_width * self.health_bar.scale
            + 2 * INDICATOR_BAR_BORDER_SIZE,
            self.game.gui_camera.viewport_height
            - self.health_bar.half_bar_height * self.health_bar.scale
            - 2 * INDICATOR_BAR_BORDER_SIZE,
        )
        self.armour_bar: IndicatorBar = IndicatorBar(
            self,
            self.game.player_gui_sprites,
            (0, 0),
            ARMOUR_INDICATOR_BAR_COLOR,
            scale=4,
        )
        self.armour_bar.position = (
            self.health_bar.center_x,
            self.health_bar.bottom
            - self.armour_bar.half_bar_height * self.armour_bar.scale,
        )
        self._entity_state.update({"money": 0.0})
        self.inventory: list[CollectibleTile] = []
        self.inventory_capacity: int = INVENTORY_WIDTH * INVENTORY_HEIGHT
        self.in_combat: bool = False
        self.left_pressed: bool = False
        self.right_pressed: bool = False
        self.up_pressed: bool = False
        self.down_pressed: bool = False

    def __repr__(self) -> str:
        return f"<Player (Position=({self.center_x}, {self.center_y}))>"

    @property
    def money(self) -> float:
        """
        Gets the player's money.

        Returns
        -------
        float
            The player's money
        """
        return self._entity_state["money"]

    @money.setter
    def money(self, value: float) -> None:
        """
        Sets the player's money.

        Parameters
        ----------
        value: float
            The new money value.
        """
        self._entity_state["money"] = value

    def _initialise_entity_state(self) -> dict[str, float]:
        """
        Initialises the entity's state dict.

        Returns
        -------
        dict[str, float]
            The initialised entity state.
        """
        return {
            "health": self.upgrade_data[0].upgrades[0].increase(0),
            "max health": self.upgrade_data[0].upgrades[0].increase(0),
            "armour": self.upgrade_data[1].upgrades[0].increase(0),
            "max armour": self.upgrade_data[1].upgrades[0].increase(0),
            "max velocity": self.upgrade_data[0].upgrades[1].increase(0),
            "armour regen cooldown": self.upgrade_data[1].upgrades[1].increase(0),
            "bonus attack cooldown": 0,
        }

    def post_on_update(self, delta_time: float = 1 / 60) -> None:
        """
        Processes player logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Check if the player can regenerate health
        if not self.in_combat:
            self.regenerate_armour(delta_time)
            if self.armour > self.max_armour:
                self.armour = self.max_armour
                logger.debug("Set player armour to max")

        # Make the player move
        self.move()

    def move(self) -> None:
        """Processes the needed actions for the entity to move."""
        # Calculate the new velocity of the player based on the keys pressed
        update_enemies = False
        force = [0, 0]
        if self.right_pressed and not self.left_pressed:
            force[0] = MOVEMENT_FORCE
        elif self.left_pressed and not self.right_pressed:
            force[0] = -MOVEMENT_FORCE
        if self.up_pressed and not self.down_pressed:
            force[1] = MOVEMENT_FORCE
        elif self.down_pressed and not self.up_pressed:
            force[1] = -MOVEMENT_FORCE
        if force != [0, 0]:
            # Apply the force
            resultant_force = (
                force[0],
                force[1],
            )
            self.physics.apply_force(self, resultant_force)
            logger.debug(f"Applied force {resultant_force} to player")

            # Set update_enemies
            update_enemies = True

        # Check if we need to update the enemy's line of sight
        line_of_sights = []
        if update_enemies:
            # Update the enemy's line of sight check
            for enemy in self.game.enemy_sprites:
                line_of_sights.append(enemy.check_line_of_sight())  # noqa

        # Check if the player is in combat
        self.in_combat = any(line_of_sights)
        if self.in_combat:
            self.time_out_of_combat = 0

    def attack(self) -> None:
        """Runs the player's current attack algorithm."""
        # Check if the player can attack
        if self.time_since_last_attack < (
            self.current_attack.attack_cooldown + self.bonus_attack_cooldown
        ):
            return

        # Reset the player's combat variables and attack
        self.time_since_armour_regen = self.armour_regen_cooldown
        self.time_since_last_attack = 0
        self.time_out_of_combat = 0
        self.in_combat = True

        # Find out what attack algorithm is selected. We also need to check if the
        # player can attack
        match type(self.current_attack):
            case AttackAlgorithmType.RANGED.value:
                self.current_attack.process_attack(self.game.bullet_sprites)
            case AttackAlgorithmType.MELEE.value:
                # # Update the framebuffer to ensure collision detection is accurate
                # self.melee_shader.update_collision()
                # result = self.melee_shader.run_shader()
                result = []
                for enemy in self.game.enemy_sprites:
                    vec_x, vec_y = (
                        enemy.center_x - self.center_x,
                        enemy.center_y - self.center_y,
                    )
                    angle = math.degrees(math.atan2(vec_y, vec_x))
                    if angle < 0:
                        angle += 360
                    if arcade.has_line_of_sight(
                        self.position,
                        enemy.position,
                        self.game.wall_sprites,
                        self.current_attack.attack_range * SPRITE_SIZE,
                    ) and (
                        self.direction - self.player_data.melee_degree // 2
                    ) <= angle <= (
                        self.direction + self.player_data.melee_degree // 2
                    ):
                        result.append(enemy)
                self.current_attack.process_attack(result)
            case AttackAlgorithmType.AREA_OF_EFFECT.value:
                self.current_attack.process_attack(self.game.enemy_sprites)

    def add_item_to_inventory(self, item: CollectibleTile) -> bool:
        """
        Adds an item to the player's inventory.

        Parameters
        ----------
        item: CollectibleTile
            The item to add to the player's inventory.

        Returns
        -------
        bool
            Whether the add was successful or not.
        """
        # Check if the array is full
        if len(self.inventory) == self.inventory_capacity:
            logger.info(f"Cannot add item {item} to full inventory")
            return False

        # Add the item to the array
        self.inventory.append(item)

        # Update the inventory grid
        self.game.window.views["InventoryView"].update_grid()  # type: ignore

        # Add successful
        logger.info(f"Adding item {item} to inventory")
        return True
