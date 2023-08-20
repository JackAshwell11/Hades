"""Manages the various systems that can be used to manipulate components."""
from __future__ import annotations

# Builtin
import math
import random
from typing import TYPE_CHECKING, ClassVar, TypedDict

# Custom
from hades.constants import (
    ARMOUR_REGEN_AMOUNT,
    ATTACK_RANGE,
    BULLET_VELOCITY,
    DAMAGE,
    FOOTPRINT_INTERVAL,
    FOOTPRINT_LIMIT,
    MELEE_ATTACK_OFFSET_LOWER,
    MELEE_ATTACK_OFFSET_UPPER,
    TARGET_DISTANCE,
)
from hades.game_objects.base import (
    AttackAlgorithms,
    SteeringBehaviours,
    SteeringMovementState,
    SystemBase,
)
from hades.game_objects.components import (
    INV,
    Armour,
    ArmourRegen,
    ArmourRegenCooldown,
    Attacks,
    FireRatePenalty,
    Footprint,
    Health,
    Inventory,
    KeyboardMovement,
    Money,
    MovementForce,
    StatusEffect,
    SteeringMovement,
    ViewDistance,
)
from hades.game_objects.steering import (
    Vec2d,
    arrive,
    evade,
    flee,
    follow_path,
    obstacle_avoidance,
    pursuit,
    seek,
    wander,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.components import (
        GameObjectAttributeBase,
    )


__all__ = (
    "ArmourRegenSystem",
    "AttackResult",
    "FootprintSystem",
    "GameObjectAttributeError",
    "GameObjectAttributeSystem",
    "InventorySpaceError",
    "InventorySystem",
    "KeyboardMovementSystem",
    "SteeringMovementSystem",
)


class AttackResult(TypedDict, total=False):
    """Holds the result of an attack."""

    ranged_attack: tuple[Vec2d, float, float]


class GameObjectAttributeError(Exception):
    """Raised when there is an error with a game object attribute."""

    def __init__(self: GameObjectAttributeError, *, name: str, error: str) -> None:
        """Initialise the object.

        Args:
            name: The name of the game object attribute.
            error: The problem raised by the game object attribute.
        """
        super().__init__(f"The game object attribute `{name}` cannot {error}.")


class InventorySpaceError(Exception):
    """Raised when there is a space problem with the inventory."""

    def __init__(self: InventorySpaceError, *, full: bool) -> None:
        """Initialise the object.

        Args:
            full: Whether the inventory is empty or full.
        """
        super().__init__(f"The inventory is {'full' if full else 'empty'}.")


class ArmourRegenSystem(SystemBase):
    """Provides facilities to manipulate armour regen components."""

    def update(self: ArmourRegenSystem, delta_time: float) -> None:
        """Process update logic for an armour regeneration component.

        Args:
            delta_time: The time interval since the last time the function was called.
        """
        for _, (
            armour,
            armour_regen,
            armour_regen_cooldown,
        ) in self.registry.get_components(Armour, ArmourRegen, ArmourRegenCooldown):
            armour_regen.time_since_armour_regen += delta_time
            if armour_regen.time_since_armour_regen >= armour_regen_cooldown.value:
                armour.value += ARMOUR_REGEN_AMOUNT
                armour_regen.time_since_armour_regen = 0


class AttackSystem(SystemBase):
    """Provides facilities to manipulate attack components."""

    def _area_of_effect_attack(
        self: AttackSystem,
        current_position: Vec2d,
        targets: list[int],
    ) -> None:
        """Perform an area of effect attack around the game object.

        Args:
            current_position: The current position of the game object.
            targets: The targets to attack.
        """
        # Find all targets that are within range and attack them
        for target in targets:
            if (
                current_position.get_distance_to(
                    self.registry.get_physics_object_for_game_object(target).position,
                )
                <= ATTACK_RANGE
            ):
                self.registry.get_system(GameObjectAttributeSystem).deal_damage(
                    target,
                    DAMAGE,
                )

    def _melee_attack(
        self: AttackSystem,
        current_position: Vec2d,
        current_rotation: float,
        targets: list[int],
    ) -> None:
        """Perform a melee attack in the direction the game object is facing.

        Args:
            current_position: The current position of the game object.
            current_rotation: The current rotation of the game object in radians.
            targets: The targets to attack.
        """
        # Calculate a vector that is perpendicular to the current rotation of the game
        # object
        rotation = Vec2d(
            math.sin(current_rotation),
            math.cos(current_rotation),
        )

        # Find all targets that can be attacked
        for target in targets:
            # Calculate the angle between the current rotation of the game object and
            # the direction the target is in
            target_position = self.registry.get_physics_object_for_game_object(
                target,
            ).position
            theta = (target_position - current_position).get_angle_between(rotation)

            # Test if the target is within range and within the circle's sector
            if (
                current_position.get_distance_to(target_position) <= ATTACK_RANGE
                and theta <= MELEE_ATTACK_OFFSET_LOWER
                or theta >= MELEE_ATTACK_OFFSET_UPPER
            ):
                self.registry.get_system(GameObjectAttributeSystem).deal_damage(
                    target,
                    DAMAGE,
                )

    def _ranged_attack(
        self: AttackSystem,
        current_position: Vec2d,
        current_rotation: float,
    ) -> AttackResult:
        """Perform a ranged attack in the direction the game object is facing.

        Args:
            current_position: The current position of the game object.
            current_rotation: The current rotation of the game object in radians.

        Returns:
            The result of the attack.
        """
        # Calculate the bullet's angle of rotation
        angle_radians = current_rotation * math.pi / 180

        # Return the result of the attack
        return {
            "ranged_attack": (
                current_position,
                math.cos(angle_radians) * BULLET_VELOCITY,
                math.sin(angle_radians) * BULLET_VELOCITY,
            ),
        }

    def do_attack(
        self: AttackSystem,
        game_object_id: int,
        targets: list[int],
    ) -> AttackResult:
        """Perform the currently selected attack algorithm.

        Args:
            game_object_id: The ID of the game object to perform the attack for.
            targets: The targets to attack.

        Returns:
            The result of the attack.
        """
        # Perform the attack on the targets
        attacks, current_physics = self.registry.get_component_for_game_object(
            game_object_id,
            Attacks,
        ), self.registry.get_physics_object_for_game_object(game_object_id)
        match attacks.attacks[attacks.attack_state]:
            case AttackAlgorithms.AREA_OF_EFFECT_ATTACK:
                self._area_of_effect_attack(current_physics.position, targets)
            case AttackAlgorithms.MELEE_ATTACK:
                self._melee_attack(
                    current_physics.position,
                    math.radians(current_physics.rotation),
                    targets,
                )
            case AttackAlgorithms.RANGED_ATTACK:
                return self._ranged_attack(
                    current_physics.position,
                    math.radians(current_physics.rotation),
                )
            case _:  # pragma: no cover
                # This should never happen as all attacks are covered above
                raise ValueError

        # Return an empty result as no ranged attack was performed
        return {}

    def previous_attack(self: AttackSystem, game_object_id: int) -> None:
        """Select the previous attack algorithm.

        Args:
            game_object_id: The ID of the game object to select the previous attack for.
        """
        attacks = self.registry.get_component_for_game_object(game_object_id, Attacks)
        attacks.attack_state = max(attacks.attack_state - 1, 0)

    def next_attack(self: AttackSystem, game_object_id: int) -> None:
        """Select the next attack algorithm.

        Args:
            game_object_id: The ID of the game object to select the previous attack for.
        """
        attacks = self.registry.get_component_for_game_object(game_object_id, Attacks)
        attacks.attack_state = min(attacks.attack_state + 1, len(attacks.attacks) - 1)


class FootprintSystem(SystemBase):
    """Provides facilities to manipulate footprint components."""

    def on_update(self: FootprintSystem, delta_time: float) -> None:
        """Process update logic for a footprint component.

        Args:
            delta_time: The time interval since the last time the function was called.
        """
        for game_object_id, footprint in self.registry.get_components(Footprint):
            # Update the time since the last footprint then check if a new footprint
            # should be created
            footprint.time_since_last_footprint += delta_time
            if footprint.time_since_last_footprint < FOOTPRINT_INTERVAL:
                return

            # Reset the counter and create a new footprint making sure to only keep
            # FOOTPRINT_LIMIT footprints
            current_position = self.registry.get_physics_object_for_game_object(
                game_object_id,
            ).position
            footprint.time_since_last_footprint = 0
            if len(footprint.footprints) >= FOOTPRINT_LIMIT:
                footprint.footprints.pop(0)
            footprint.footprints.append(current_position)

            # Update the path list for all SteeringMovement components
            self.registry.get_system(SteeringMovementSystem).update_path_list(
                game_object_id,
                footprint.footprints,
            )


class GameObjectAttributeSystem(SystemBase):
    """Provides facilities to manipulate game object attributes."""

    GAME_OBJECT_ATTRIBUTES: ClassVar[set[type[GameObjectAttributeBase]]] = {
        Armour,
        ArmourRegenCooldown,
        FireRatePenalty,
        Health,
        Money,
        MovementForce,
        ViewDistance,
    }

    def update(self: GameObjectAttributeSystem, delta_time: float) -> None:
        """Process update logic for a game object attribute.

        Args:
            delta_time: The time since the last update.
        """
        # Loop over all game object attributes and update them
        for game_object_attribute_type in self.GAME_OBJECT_ATTRIBUTES:
            for _, game_object_attribute in self.registry.get_components(
                game_object_attribute_type,
            ):
                # Update the status effect if one is applied
                if status_effect := game_object_attribute.applied_status_effect:
                    status_effect.time_counter += delta_time
                    if status_effect.time_counter >= status_effect.duration:
                        game_object_attribute.value = min(
                            game_object_attribute.value,
                            status_effect.original_value,
                        )
                        game_object_attribute.max_value = (
                            status_effect.original_max_value
                        )
                        game_object_attribute.applied_status_effect = None

    def upgrade(
        self: GameObjectAttributeSystem,
        game_object_id: int,
        game_object_attribute_type: type[GameObjectAttributeBase],
        increase: Callable[[int], float],
    ) -> bool:
        """Upgrade the game object attribute to the next level if possible.

        Args:
            game_object_id: The ID of the game object to upgrade the game object
            attribute for.
            game_object_attribute_type: The type of game object attribute to upgrade.
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.

        Returns:
            Whether the game object attribute upgrade was successful or not.

        Raises:
            GameObjectAttributeError: The game object attribute cannot be upgraded.
        """
        # Check if the attribute can be upgraded
        game_object_attribute = self.registry.get_component_for_game_object(
            game_object_id,
            game_object_attribute_type,
        )
        if not game_object_attribute.upgradable:
            raise GameObjectAttributeError(
                name=game_object_attribute.__class__.__name__,
                error="be upgraded",
            )

        # Check if the current level is below the level limit
        if game_object_attribute.current_level >= game_object_attribute.level_limit:
            return False

        # Upgrade the attribute based on the difference between the current level and
        # the next
        diff = increase(game_object_attribute.current_level + 1) - increase(
            game_object_attribute.current_level,
        )
        game_object_attribute.max_value += diff
        game_object_attribute.current_level += 1
        game_object_attribute.value += diff
        return True

    def apply_instant_effect(
        self: GameObjectAttributeSystem,
        game_object_id: int,
        game_object_attribute_type: type[GameObjectAttributeBase],
        increase: Callable[[int], float],
        level: int,
    ) -> bool:
        """Apply an instant effect to the game object attribute if possible.

        Args:
            game_object_id: The game object ID to upgrade the attribute of.
            game_object_attribute_type: The type of game object attribute to apply an
            instant effect to.
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.
            level: The level to initialise the instant effect at.

        Returns:
            Whether the instant effect could be applied or not.

        Raises:
            GameObjectAttributeError: The game object attribute cannot have an instant
            effect.
        """
        # Check if the attribute can have an instant effect
        game_object_attribute = self.registry.get_component_for_game_object(
            game_object_id,
            game_object_attribute_type,
        )
        if not game_object_attribute.instant_effect:
            raise GameObjectAttributeError(
                name=game_object_attribute.__class__.__name__,
                error="have an instant effect",
            )

        # Check if the attribute's value is already at max
        if game_object_attribute.value == game_object_attribute.max_value:
            return False

        # Add the instant effect to the attribute
        game_object_attribute.value += increase(level)
        return True

    def apply_status_effect(
        self: GameObjectAttributeSystem,
        game_object_id: int,
        game_object_attribute_type: type[GameObjectAttributeBase],
        status_effect: tuple[Callable[[int], float], Callable[[int], float]],
        level: int,
    ) -> bool:
        """Apply a status effect to the attribute if possible.

        Args:
            game_object_id: The game object ID to upgrade the attribute of.
            game_object_attribute_type: The type of game object attribute to apply a
                status effect to.
            status_effect: The exponential lambda functions which calculate the next
                level's value and duration based on the current level.
            level: The level to initialise the status effect at.

        Returns:
            Whether the status effect could be applied or not.

        Raises:
            GameObjectAttributeError: The game object attribute cannot have a status
            effect.
        """
        # Check if the attribute can have a status effect
        game_object_attribute = self.registry.get_component_for_game_object(
            game_object_id,
            game_object_attribute_type,
        )
        if not game_object_attribute.status_effect:
            raise GameObjectAttributeError(
                name=game_object_attribute.__class__.__name__,
                error="have a status effect",
            )

        # Check if the attribute already has a status effect applied
        if game_object_attribute.applied_status_effect:
            return False

        # Apply the status effect to this attribute
        increase, duration = status_effect
        game_object_attribute.applied_status_effect = StatusEffect(
            increase(level),
            duration(level),
            game_object_attribute.value,
            game_object_attribute.max_value,
        )
        game_object_attribute.value += game_object_attribute.applied_status_effect.value
        game_object_attribute.max_value += (
            game_object_attribute.applied_status_effect.value
        )
        return True

    def deal_damage(
        self: GameObjectAttributeSystem,
        game_object_id: int,
        damage: int,
    ) -> None:
        """Deal damage to the game object.

        Args:
            game_object_id: The game object ID.
            damage: The damage that should be dealt to the game object.
        """
        # Damage the armour and carry over the extra damage to the health
        health, armour = (
            self.registry.get_component_for_game_object(game_object_id, Health),
            self.registry.get_component_for_game_object(game_object_id, Armour),
        )
        health.value -= max(damage - armour.value, 0)
        armour.value -= damage


class InventorySystem(SystemBase):
    """Provides facilities to manipulate inventory components."""

    def add_item_to_inventory(
        self: InventorySystem,
        game_object_id: int,
        item: INV,
    ) -> None:
        """Add an item to the inventory.

        Args:
            game_object_id: The ID of the game object to add the item to.
            item: The item to add to the inventory.

        Raises:
            InventorySpaceError: The inventory is full.
        """
        inventory = self.registry.get_component_for_game_object(
            game_object_id,
            Inventory,
        )
        if len(inventory.inventory) == inventory.width * inventory.height:
            raise InventorySpaceError(full=True)
        inventory.inventory.append(item)

    def remove_item_from_inventory(
        self: InventorySystem,
        game_object_id: int,
        index: int,
    ) -> INV:
        """Remove an item at a specific index.

        Args:
            game_object_id: The ID of the game object to remove an item from.
            index: The index to remove an item at.

        Returns:
            The item at position `index` in the inventory.

        Raises:
            InventorySpaceError: The inventory is empty.
        """
        inventory = self.registry.get_component_for_game_object(
            game_object_id,
            Inventory,
        )
        if len(inventory.inventory) < index:
            raise InventorySpaceError(full=False)
        return inventory.inventory.pop(index)


class KeyboardMovementSystem(SystemBase):
    """Provides facilities to manipulate keyboard movement components."""

    def calculate_force(self: KeyboardMovementSystem, game_object_id: int) -> Vec2d:
        """Calculate the new force to apply to the game object.

        Args:
            game_object_id: The ID of the game object to calculate force for.

        Returns:
            The new force to apply to the game object.
        """
        keyboard_movement = self.registry.get_component_for_game_object(
            game_object_id,
            KeyboardMovement,
        )
        return (
            Vec2d(
                keyboard_movement.east_pressed - keyboard_movement.west_pressed,
                keyboard_movement.north_pressed - keyboard_movement.south_pressed,
            )
            * self.registry.get_component_for_game_object(
                game_object_id,
                MovementForce,
            ).value
        )


class SteeringMovementSystem(SystemBase):
    """Provides facilities to manipulate steering movement components."""

    def calculate_force(self: SteeringMovementSystem, game_object_id: int) -> Vec2d:
        """Calculate the new force to apply to the game object.

        Args:
            game_object_id: The ID of the game object to calculate force for.

        Returns:
            The new force to apply to the game object.
        """
        # Determine if the movement state should change or not
        steering_movement, current_physics = (
            self.registry.get_component_for_game_object(
                game_object_id,
                SteeringMovement,
            ),
            self.registry.get_physics_object_for_game_object(game_object_id),
        )
        target_physics = self.registry.get_physics_object_for_game_object(
            steering_movement.target_id,
        )
        if (
            current_physics.position.get_distance_to(target_physics.position)
            <= TARGET_DISTANCE
        ):
            steering_movement.movement_state = SteeringMovementState.TARGET
        elif steering_movement.path_list:
            steering_movement.movement_state = SteeringMovementState.FOOTPRINT
        else:
            steering_movement.movement_state = SteeringMovementState.DEFAULT

        # Calculate the new force to apply to the game object
        steering_force = Vec2d(0, 0)
        for behaviour in steering_movement.behaviours.get(
            steering_movement.movement_state,
            [],
        ):
            match behaviour:
                case SteeringBehaviours.ARRIVE:
                    steering_force += arrive(
                        current_physics.position,
                        target_physics.position,
                    )
                case SteeringBehaviours.EVADE:
                    steering_force += evade(
                        current_physics.position,
                        target_physics.position,
                        target_physics.velocity,
                    )
                case SteeringBehaviours.FLEE:
                    steering_force += flee(
                        current_physics.position,
                        target_physics.position,
                    )
                case SteeringBehaviours.FOLLOW_PATH:
                    steering_force += follow_path(
                        current_physics.position,
                        steering_movement.path_list,
                    )
                case SteeringBehaviours.OBSTACLE_AVOIDANCE:
                    steering_force += obstacle_avoidance(
                        current_physics.position,
                        current_physics.velocity,
                        steering_movement.walls,
                    )
                case SteeringBehaviours.PURSUIT:
                    steering_force += pursuit(
                        current_physics.position,
                        target_physics.position,
                        target_physics.velocity,
                    )
                case SteeringBehaviours.SEEK:
                    steering_force += seek(
                        current_physics.position,
                        target_physics.position,
                    )
                case SteeringBehaviours.WANDER:
                    steering_force += wander(
                        current_physics.velocity,
                        random.randint(0, 360),
                    )
                case _:  # pragma: no cover
                    # This should never happen as all behaviours are covered above
                    raise ValueError
        return (
            steering_force.normalised()
            * self.registry.get_component_for_game_object(
                game_object_id,
                MovementForce,
            ).value
        )

    def update_path_list(
        self: SteeringMovementSystem,
        game_object_id: int,
        footprints: list[Vec2d],
    ) -> None:
        """Update the path list for the game object to follow.

        Args:
            game_object_id: The ID of the game object to update the path list for.
            footprints: The list of footprints to follow.
        """
        # Get the closest footprint to the target and test if one exists
        steering_movement, current_position = (
            self.registry.get_component_for_game_object(
                game_object_id,
                SteeringMovement,
            ),
            self.registry.get_physics_object_for_game_object(game_object_id).position,
        )
        closest_footprints = [
            footprint
            for footprint in footprints
            if current_position.get_distance_to(footprint) <= TARGET_DISTANCE
        ]
        if not closest_footprints:
            steering_movement.path_list.clear()
            return

        # Get the closest footprint to the target and start following the footprints
        # from that footprint
        target_footprint = min(
            closest_footprints,
            key=self.registry.get_physics_object_for_game_object(
                steering_movement.target_id,
            ).position.get_distance_to,
        )
        steering_movement.path_list = footprints[footprints.index(target_footprint) :]
