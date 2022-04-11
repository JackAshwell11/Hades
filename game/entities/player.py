from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from constants.entity import PLAYER, AttackAlgorithmType, EntityID
from constants.general import INVENTORY_HEIGHT, INVENTORY_WIDTH
from entities.base import Entity
from entities.status_effect import StatusEffect
from melee_shader import MeleeShader

if TYPE_CHECKING:
    from constants.entity import BaseType
    from entities.base import Item
    from views.game import Game

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

    Attributes
    ----------
    melee_shader: MeleeShader
        The OpenGL shader used to find and attack any enemies within a specific distance
        around the player based on their direction.
    inventory: list[Item]
        The list which stores the player's inventory.
    inventory_capacity: int
        The total capacity of the inventory.
    applied_effects: list[StatusEffect]
        The currently applied status effects.
    in_combat: bool
        Whether the player is in combat or not.
    """

    # Class variables
    entity_id: EntityID = EntityID.PLAYER
    entity_data: BaseType = PLAYER

    def __init__(self, game: Game, x: int, y: int) -> None:
        super().__init__(game, x, y)
        self.melee_shader: MeleeShader = MeleeShader(self.game)
        self.inventory: list[Item] = []
        self.inventory_capacity: int = INVENTORY_WIDTH * INVENTORY_HEIGHT
        self.applied_effects: list[StatusEffect] = []
        self.state_modifiers: dict[str, int] = {
            "bonus health": 0,
            "bonus armour": 0,
        }
        self.in_combat: bool = False

    def __repr__(self) -> str:
        return f"<Player (Position=({self.center_x}, {self.center_y}))>"

    def on_update(self, delta_time: float = 1 / 60) -> None:
        """
        Processes player logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the player's time since last attack
        self.time_since_last_attack += delta_time

        # Check if the player can regenerate health
        if not self.in_combat:
            self.check_armour_regen(delta_time)
            armour_limit = (
                self.entity_type.armour + self.state_modifiers["bonus armour"]
            )
            self.armour: int  # Mypy gives self.armour an undetermined type error
            if self.armour > armour_limit:
                self.armour = armour_limit
                logger.debug("Set player armour to max")

        # Update any status effects
        for status_effect in self.applied_effects:
            logger.debug(f"Updating status effect {status_effect}")
            status_effect.update(delta_time)

    def add_item_to_inventory(self, item: Item) -> bool:
        """
        Adds an item to the player's inventory.

        Parameters
        ----------
        item: Item
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
        self.game.window.views["InventoryView"].update_grid()

        # Add successful
        logger.info(f"Adding item {item} to inventory")
        return True

    def add_status_effect(self, effect: StatusEffect) -> None:
        """
        Adds a status effect to the player

        Parameters
        ----------
        effect: StatusEffect
            The status effect to add to the player.
        """
        self.applied_effects.append(effect)
        effect.apply_effect()
        logger.info(f"Adding effect {effect} to player")

    def attack(self) -> None:
        """Runs the player's current attack algorithm."""
        # Find out what attack algorithm is selected
        match type(self.current_attack):
            case AttackAlgorithmType.RANGED.value:
                self.current_attack.process_attack(self.game.bullet_sprites)
            case AttackAlgorithmType.MELEE.value:
                # Update the framebuffer to ensure collision detection is accurate
                self.melee_shader.update_collision()
                self.current_attack.process_attack(self.melee_shader.run_shader())
            case AttackAlgorithmType.AREA_OF_EFFECT.value:
                self.current_attack.process_attack(self.game.enemies)
