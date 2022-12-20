"""Manages the player object and its various functions."""
from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from hades.constants.game_objects import (
    ARMOUR_INDICATOR_BAR_COLOR,
    HEALTH_INDICATOR_BAR_COLOR,
    INDICATOR_BAR_BORDER_SIZE,
    MOVEMENT_FORCE,
    SPRITE_SIZE,
    AttackAlgorithmType,
    EntityAttributeType,
    ObjectID,
)
from hades.constants.general import INVENTORY_HEIGHT, INVENTORY_WIDTH
from hades.game_objects.attributes import EntityAttribute, UpgradablePlayerSection
from hades.game_objects.base import Entity, IndicatorBar
from hades.melee_shader import MeleeShader

if TYPE_CHECKING:
    from hades.constants.game_objects import BaseData, PlayerData
    from hades.game_objects.base import CollectibleTile
    from hades.views.game import Game

__all__ = ("Player",)

# Get the logger
logger = logging.getLogger(__name__)


class Player(Entity):
    """Represents the player character in the game.

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
    upgrade_sections: list[UpgradablePlayerSection]
        Stores the upgradable section objects for each of the player's upgrade sections.
    inventory: list[Tile]
        The list which stores the player's inventory.
    inventory_capacity: int
        The total capacity of the inventory.
    current_tile_pos: tuple[int, int]
        The player's current tile position in the vector field.
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
    object_id: ObjectID = ObjectID.PLAYER

    def __init__(self, game: Game, x: int, y: int, player_type: BaseData) -> None:
        super().__init__(game, x, y, player_type)
        self.melee_shader: MeleeShader = MeleeShader(self.game)
        self.upgrade_sections: list[UpgradablePlayerSection] = [
            UpgradablePlayerSection(self, section_type, cost_function, 0)
            for (
                section_type,
                cost_function,
            ) in self.player_data.section_upgrade_data.items()
        ]
        self.health_bar: IndicatorBar = IndicatorBar(
            self,
            self.game.player_gui_sprites,
            (0, 0),
            HEALTH_INDICATOR_BAR_COLOR,
            border_size=INDICATOR_BAR_BORDER_SIZE,
            scale=4,
        )
        self.health_bar.position = (
            (self.health_bar.bar_width / 2) * self.health_bar.scale
            + 2 * INDICATOR_BAR_BORDER_SIZE,
            self.game.gui_camera.viewport_height
            - (self.health_bar.bar_height / 2) * self.health_bar.scale
            - 2 * INDICATOR_BAR_BORDER_SIZE,
        )
        self.armour_bar: IndicatorBar = IndicatorBar(
            self,
            self.game.player_gui_sprites,
            (0, 0),
            ARMOUR_INDICATOR_BAR_COLOR,
            border_size=INDICATOR_BAR_BORDER_SIZE,
            scale=4,
        )
        self.armour_bar.position = (
            self.health_bar.center_x,
            self.health_bar.bottom
            - (self.health_bar.bar_height / 2) * self.armour_bar.scale,
        )
        self.inventory: list[CollectibleTile] = []
        self.inventory_capacity: int = INVENTORY_WIDTH * INVENTORY_HEIGHT
        self.current_tile_pos: tuple[int, int] = (-1, -1)
        self.in_combat: bool = False
        self.left_pressed: bool = False
        self.right_pressed: bool = False
        self.up_pressed: bool = False
        self.down_pressed: bool = False

    @property
    def player_data(self) -> PlayerData:
        """Get the player data if it exists.

        Returns
        -------
        PlayerData
            The player data.
        """
        # Make sure the entity type is valid
        assert self.entity_type.player_data is not None

        # Return the player data
        return self.entity_type.player_data

    @property
    def money(self) -> EntityAttribute:
        """Get the player's money.

        Returns
        -------
        EntityAttribute
            The player's money
        """
        return self.entity_state[EntityAttributeType.MONEY]

    def _initialise_entity_state(self) -> dict[EntityAttributeType, EntityAttribute]:
        """Initialise the player's state.

        Returns
        -------
        dict[EntityAttributeType, EntityAttribute]
            The initialised entity state.
        """
        return {
            attribute_type: EntityAttribute(self, attribute_data, 0)
            for attribute_type, attribute_data in self.attribute_data.items()
        }

    def post_on_update(self, delta_time: float) -> None:
        """Process custom player logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Make sure variables needed are valid
        assert self.game.vector_field is not None

        # Check if the player can regenerate health
        if not self.in_combat:
            self.regenerate_armour(delta_time)

        # Check if the player has moved tile positions
        new_tile_pos = self.game.vector_field.pixel_to_grid_pos(self.position)
        if self.current_tile_pos != new_tile_pos:
            # Player has moved tile positions so update vector field
            self.current_tile_pos = new_tile_pos
            self.game.possible_enemy_spawns = self.game.vector_field.recalculate_map(
                self.position, self.entity_data.view_distance
            )

        # Make the player move
        self.move(delta_time)

    def move(self, delta_time: float) -> None:
        """Process the needed actions for the player to move.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Calculate the force to apply to the player based on the keys pressed
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

        # Check if the player is in combat
        self.in_combat = any(
            enemy.player_within_range for enemy in self.game.enemy_sprites  # noqa
        )
        if self.in_combat:
            self.time_out_of_combat = 0

    def attack(self) -> None:
        """Run the player's current attack algorithm."""
        # Check if the player can attack
        if self.time_since_last_attack < (
            self.current_attack.attack_data.attack_cooldown
            * self.fire_rate_penalty.value
        ):
            return

        # Reset the player's combat variables and attack
        self.time_since_armour_regen = self.armour_regen_cooldown.value
        self.time_since_last_attack = 0
        self.time_out_of_combat = 0
        self.in_combat = True

        # Find out what attack algorithm is selected. We also need to check if the
        # player can attack or not
        if self.current_attack.attack_type is AttackAlgorithmType.RANGED:
            self.current_attack.process_attack(
                target_spritelist=self.game.bullet_sprites
            )
        elif self.current_attack.attack_type is AttackAlgorithmType.MELEE:
            # # Update the framebuffer to ensure collision detection is accurate
            # self.melee_shader.update_collision()
            # result = self.melee_shader.run_shader()
            result = []
            for enemy in self.game.enemy_sprites:  # type: Entity
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
                    self.current_attack.attack_data.attack_range * SPRITE_SIZE,
                ) and (
                    self.direction - self.player_data.melee_degree // 2
                ) <= angle <= (
                    self.direction + self.player_data.melee_degree // 2
                ):
                    result.append(enemy)
            self.current_attack.process_attack(target_entities=result)
        elif self.current_attack.attack_type is AttackAlgorithmType.AREA_OF_EFFECT:
            self.current_attack.process_attack(
                target_spritelist=self.game.enemy_sprites
            )

    def add_item_to_inventory(self, item: CollectibleTile) -> bool:
        """Add an item to the player's inventory.

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
            logger.info("Cannot add item %r to full inventory", item)
            return False

        # Add the item to the array
        self.inventory.append(item)

        # Update the inventory grid
        self.game.window.views["InventoryView"].update_grid()  # type: ignore

        # Add successful
        logger.info("Adding item %r to inventory", item)
        return True

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<Player (Position=({self.center_x}, {self.center_y}))>"
