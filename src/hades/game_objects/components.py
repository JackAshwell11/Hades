"""Manages various components available to the game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Generic, TypeVar, cast

# Custom
from hades.constants import ARMOUR_REGEN_AMOUNT, FOOTPRINT_INTERVAL, FOOTPRINT_LIMIT
from hades.game_objects.attributes import Armour, ArmourRegenCooldown
from hades.game_objects.base import ComponentType, GameObjectComponent
from hades.game_objects.movements import MovementBase, SteeringMovement

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData, Vec2d
    from hades.game_objects.movements import PhysicsObject
    from hades.game_objects.system import ECS

__all__ = (
    "ArmourRegen",
    "Footprint",
    "InstantEffects",
    "Inventory",
    "InventorySpaceError",
    "StatusEffects",
)

# Define a generic type for the inventory
T = TypeVar("T")


class InventorySpaceError(Exception):
    """Raised when there is a space problem with the inventory."""

    def __init__(self: InventorySpaceError, *, full: bool) -> None:
        """Initialise the object.

        Args:
            full: Whether the inventory is empty or full.
        """
        super().__init__(f"The inventory is {'full' if full else 'empty'}.")


class ArmourRegen(GameObjectComponent):
    """Allows a game object to regenerate armour.

    Attributes:
        armour: The game object's armour component.
        armour_regen_cooldown: The game object's armour regen cooldown component.
        time_since_armour_regen: The time since the game object last regenerated armour.
    """

    __slots__ = ("armour", "armour_regen_cooldown", "time_since_armour_regen")

    # Class variables
    component_type: ComponentType = ComponentType.ARMOUR_REGEN

    def __init__(
        self: ArmourRegen,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.armour: Armour = cast(
            Armour,
            self.system.get_component_for_game_object(
                self.game_object_id,
                ComponentType.ARMOUR,
            ),
        )
        self.armour_regen_cooldown: ArmourRegenCooldown = cast(
            ArmourRegenCooldown,
            self.system.get_component_for_game_object(
                self.game_object_id,
                ComponentType.ARMOUR_REGEN_COOLDOWN,
            ),
        )
        self.time_since_armour_regen: float = 0

    def on_update(self: ArmourRegen, delta_time: float) -> None:
        """Process armour regeneration update logic.

        Args:
            delta_time: Time interval since the last time the function was called.
        """
        self.time_since_armour_regen += delta_time
        if self.time_since_armour_regen >= self.armour_regen_cooldown.value:
            self.armour.value += ARMOUR_REGEN_AMOUNT
            self.time_since_armour_regen = 0

    def __repr__(self: ArmourRegen) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<ArmourRegen (Time since armour regen={self.time_since_armour_regen})>"


class Footprint(GameObjectComponent):
    """Allows a game object to periodically leave footprints around the game map.

    Attributes:
        footprints: The footprints created by the game object.
        physics_object: The physics object for the game object.
        time_since_last_footprint: The time since the game object last left a footprint.
    """

    __slots__ = ("footprints", "physics_object", "time_since_last_footprint")

    # Class variables
    component_type: ComponentType = ComponentType.FOOTPRINT

    def __init__(
        self: Footprint,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.footprints: list[Vec2d] = []
        self.physics_object: PhysicsObject = (
            self.system.get_physics_object_for_game_object(self.game_object_id)
        )
        self.time_since_last_footprint: float = 0

    def on_update(self: Footprint, delta_time: float) -> None:
        """Process AI update logic.

        Args:
            delta_time: Time interval since the last time the function was called.
        """
        # Update the time since the last footprint then check if a new footprint should
        # be created
        self.time_since_last_footprint += delta_time
        if self.time_since_last_footprint < FOOTPRINT_INTERVAL:
            return

        # Reset the counter and create a new footprint making sure to only keep
        # FOOTPRINT_LIMIT footprints
        self.time_since_last_footprint = 0
        if len(self.footprints) >= FOOTPRINT_LIMIT:
            self.footprints.pop(0)
        self.footprints.append(self.physics_object.position)

        # Update the path list for all SteeringMovement components
        for movement_component in self.system.get_components_for_component_type(
            ComponentType.MOVEMENTS,
        ):
            if not cast(MovementBase, movement_component).is_player_controlled:
                cast(SteeringMovement, movement_component).update_path_list(
                    self.footprints,
                )

    def __repr__(self: Footprint) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<Footprint (Footprint count={len(self.footprints)}) (Time since last"
            f" footprint={self.time_since_last_footprint})>"
        )


class InstantEffects(GameObjectComponent):
    """Allows a game object to provide instant effects.

    Attributes:
        level_limit: The level limit of the instant effects.
        instant_effects: The instant effects provided by the game object.
    """

    __slots__ = ("instant_effects", "level_limit")

    # Class variables
    component_type: ComponentType = ComponentType.INSTANT_EFFECTS

    def __init__(
        self: InstantEffects,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.level_limit, self.instant_effects = component_data["instant_effects"]

    def __repr__(self: InstantEffects) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<InstantEffects (Level limit={self.level_limit})>"


class Inventory(Generic[T], GameObjectComponent):
    """Allows a game object to have a fixed size inventory.

    Attributes:
        width: The width of the inventory.
        height: The height of the inventory.
        inventory: The game object's inventory.
    """

    __slots__ = (
        "width",
        "height",
        "inventory",
    )

    # Class variables
    component_type: ComponentType = ComponentType.INVENTORY

    def __init__(
        self: Inventory[T],
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.width, self.height = component_data["inventory_size"]
        self.inventory: list[T] = []

    def add_item_to_inventory(self: Inventory[T], item: T) -> None:
        """Add an item to the inventory.

        Args:
            item: The item to add to the inventory.

        Raises:
            InventorySpaceError: The inventory is full.
        """
        if len(self.inventory) == self.width * self.height:
            raise InventorySpaceError(full=True)
        self.inventory.append(item)

    def remove_item_from_inventory(self: Inventory[T], index: int) -> T:
        """Remove an item at a specific index.

        Args:
            index: The index to remove an item at.

        Returns:
            The item at position `index` in the inventory.

        Raises:
            InventorySpaceError: The inventory is empty.
        """
        if len(self.inventory) < index:
            raise InventorySpaceError(full=False)
        return self.inventory.pop(index)

    def __repr__(self: Inventory[T]) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Inventory (Width={self.width}) (Height={self.height})>"


class StatusEffects(GameObjectComponent):
    """Allows a game object to provide status effects.

    Attributes:
        level_limit: The level limit of the status effects.
        status_effects: The status effects provided by the game object.
    """

    __slots__ = ("level_limit", "status_effects")

    # Class variables
    component_type: ComponentType = ComponentType.STATUS_EFFECTS

    def __init__(
        self: StatusEffects,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.level_limit, self.status_effects = component_data["status_effects"]

    def __repr__(self: StatusEffects) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<StatusEffects (Level limit={self.level_limit})>"
