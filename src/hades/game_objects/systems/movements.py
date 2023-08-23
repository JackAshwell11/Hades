"""Manages the movements system and its various movement algorithms."""
from __future__ import annotations

# Builtin
import random

# Custom
from hades.constants import FOOTPRINT_INTERVAL, FOOTPRINT_LIMIT, TARGET_DISTANCE
from hades.game_objects.base import (
    SteeringBehaviours,
    SteeringMovementState,
    SystemBase,
)
from hades.game_objects.components import (
    Footprint,
    KeyboardMovement,
    MovementForce,
    SteeringMovement,
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

__all__ = ("KeyboardMovementSystem", "SteeringMovementSystem", "FootprintSystem")


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
        steering_movement, kinematic_owner = (
            self.registry.get_component_for_game_object(
                game_object_id,
                SteeringMovement,
            ),
            self.registry.get_kinematic_object_for_game_object(game_object_id),
        )
        kinematic_target = self.registry.get_kinematic_object_for_game_object(
            steering_movement.target_id,
        )
        if (
            kinematic_owner.position.get_distance_to(kinematic_target.position)
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
                        kinematic_owner.position,
                        kinematic_target.position,
                    )
                case SteeringBehaviours.EVADE:
                    steering_force += evade(
                        kinematic_owner.position,
                        kinematic_target.position,
                        kinematic_target.velocity,
                    )
                case SteeringBehaviours.FLEE:
                    steering_force += flee(
                        kinematic_owner.position,
                        kinematic_target.position,
                    )
                case SteeringBehaviours.FOLLOW_PATH:
                    steering_force += follow_path(
                        kinematic_owner.position,
                        steering_movement.path_list,
                    )
                case SteeringBehaviours.OBSTACLE_AVOIDANCE:
                    steering_force += obstacle_avoidance(
                        kinematic_owner.position,
                        kinematic_owner.velocity,
                        self.registry.walls,
                    )
                case SteeringBehaviours.PURSUIT:
                    steering_force += pursuit(
                        kinematic_owner.position,
                        kinematic_target.position,
                        kinematic_target.velocity,
                    )
                case SteeringBehaviours.SEEK:
                    steering_force += seek(
                        kinematic_owner.position,
                        kinematic_target.position,
                    )
                case SteeringBehaviours.WANDER:
                    steering_force += wander(
                        kinematic_owner.velocity,
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
        target_game_object_id: int,
        footprints: list[Vec2d],
    ) -> None:
        """Update the path lists for the game objects to follow.

        Args:
            target_game_object_id: The ID of the game object to update the path lists
                for.
            footprints: The list of footprints to follow.
        """
        # Update the path list for all SteeringMovement components that have the correct
        # target ID
        for game_object_id, (steering_movement,) in self.registry.get_components(
            SteeringMovement,
        ):
            if steering_movement.target_id != target_game_object_id:
                continue

            # Get the closest footprint to the target and test if one exists
            current_position = self.registry.get_kinematic_object_for_game_object(
                game_object_id,
            ).position
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
                key=self.registry.get_kinematic_object_for_game_object(
                    steering_movement.target_id,
                ).position.get_distance_to,
            )
            steering_movement.path_list = footprints[
                footprints.index(target_footprint) :
            ]


class FootprintSystem(SystemBase):
    """Provides facilities to manipulate footprint components."""

    def update(self: FootprintSystem, delta_time: float) -> None:
        """Process update logic for a footprint component.

        Args:
            delta_time: The time interval since the last time the function was called.
        """
        for game_object_id, (footprint,) in self.registry.get_components(Footprint):
            # Update the time since the last footprint then check if a new footprint
            # should be created
            footprint.time_since_last_footprint += delta_time
            if footprint.time_since_last_footprint < FOOTPRINT_INTERVAL:
                return

            # Reset the counter and create a new footprint making sure to only keep
            # FOOTPRINT_LIMIT footprints
            current_position = self.registry.get_kinematic_object_for_game_object(
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
