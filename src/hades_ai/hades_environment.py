"""Manages the environment definition for the reinforcement learning agent."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Any, Final

# Pip
import numpy as np
from arcade import get_sprites_at_point, key
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from pyglet import clock

# Custom
from hades.views.game import Game
from hades_ai.capture_window import CaptureWindow
from hades_extensions.ecs import (
    SPRITE_SIZE,
    GameObjectType,
    Vec2d,
)
from hades_extensions.ecs.components import KinematicComponent
from hades_extensions.ecs.systems import PhysicsSystem

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType

__all__ = ("HadesEnvironment",)

# The maximum velocity of the player
MAX_VELOCITY: Final[float] = 600

# The safe distance to be from a wall
SAFE_DISTANCE: Final[float] = SPRITE_SIZE * 1.5

# The size of the position history
POSITION_HISTORY_SIZE: Final[int] = 25


class HadesEnvironment(Env):  # type:ignore[misc]
    """Represents the reinforcement learning environment for Hades.

    Attributes:
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        window: The window to capture the game view.
        game: The game view.
        previous_action: The previous action taken.
        position_history: The history of the player's positions.

    Action Space:
        The action space is a discrete space with the following actions:
        - 0: No action
        - 1: Move up
        - 2: Move left
        - 3: Move down
        - 4: Move right
        - 5: Move up-right
        - 6: Move up-left
        - 7: Move down-right
        - 8: Move down-left

    Observation Space:
        The observation space is a dictionary with the following observations:
        - current_position: The current position of the player.
        - current_velocity: The current velocity of the player.
        - distance_to_walls: The distances from the player to the walls in each
            direction.
        - previous_action: The previous action taken.
        - is_near_wall: A flag indicating if the player is near a wall or not.

    Info Space:
        The info space is an empty dictionary.

    Reward Function:
        The reward function is as follows:
        - Start at a neutral reward of 0.5.
        - The agent is heavily penalised for touching a wall
        - The agent is penalised for being close to a wall.
        - The agent is penalised for being stationary for too long.
        - The agent is penalised for oscillating on the same tile.
        - The agent is rewarded for moving around.
        - The agent is rewarded for being away from walls.

    Termination Conditions:
        The episode is terminated when any of the following conditions are met:
        - The agent is outside the bounds of the dungeon.
        - The agent is inside a wall.
    """

    __slots__ = (
        "_action_to_direction",
        "action_space",
        "game",
        "observation_space",
        "position_history",
        "previous_action",
        "window",
    )

    def __init__(self: HadesEnvironment) -> None:
        """Initialise the object."""
        super().__init__()

        # Define the action space and the mapping from actions to directions
        self._action_to_direction: dict[int, list[int]] = {
            0: [],
            1: [key.W],
            2: [key.A],
            3: [key.S],
            4: [key.D],
            5: [key.W, key.D],
            6: [key.W, key.A],
            7: [key.S, key.D],
            8: [key.S, key.A],
        }
        self.action_space: Discrete = Discrete(len(self._action_to_direction))

        # Define the observation space
        self.observation_space: Dict = Dict(
            {
                "current_position": Box(
                    low=0,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "current_velocity": Box(
                    low=-MAX_VELOCITY,
                    high=MAX_VELOCITY,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "distance_to_walls": Box(
                    low=0,
                    high=np.inf,
                    shape=(8,),
                    dtype=np.float32,
                ),
                "previous_action": Discrete(9),
                "is_near_wall": Discrete(2),
            },
        )

        # Store some variables for the environment
        self.window: CaptureWindow = CaptureWindow()
        self.game: Game | None = None
        self.previous_action: int = 0
        self.position_history: list[tuple[int, int]] = []

    def _get_obs(self: HadesEnvironment) -> ObsType:
        """Returns the current observation.

        Raises:
            RuntimeError: If the game is not initialised.

        Returns:
            The current observation.
        """
        # Check if the game is initialised or not
        if not self.game:
            error = "Game is not initialised."
            raise RuntimeError(error)

        # Calculate the distances to the walls
        kinematic_component = self.game.registry.get_component(
            self.game.player,
            KinematicComponent,
        )
        current_position = np.array(
            kinematic_component.get_position(),
            dtype=np.float32,
        )
        wall_distances = np.linalg.norm(
            np.array(
                [
                    wall.pos
                    for wall in self.game.registry.get_system(
                        PhysicsSystem,
                    ).get_wall_distances(Vec2d(*current_position))
                ],
            )
            - current_position,
            axis=1,
        )

        # Return the observations
        return {
            "current_position": current_position,
            "current_velocity": np.array(
                kinematic_component.get_velocity(),
                dtype=np.float32,
            ),
            "distance_to_walls": wall_distances,
            "previous_action": self.previous_action,
            "is_near_wall": np.float32(np.any(wall_distances <= SAFE_DISTANCE)),
        }

    def reset(  # type: ignore[explicit-any]
        self: HadesEnvironment,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: The seed to use for the environment.
            options: The options to use for the environment.

        Returns:
            The initial observation and information.
        """
        # Make sure the seeding is correctly set up
        super().reset(seed=seed, options=options)

        # Reset the window and store the required variables
        self.previous_action = 0
        self.game = Game(0)
        self.position_history.clear()

        # Show the game view so that the AI agent can interact with it
        self.window.show_view(self.game)

        # Return the initial observation and information
        return self._get_obs(), {}

    def step(  # type: ignore[explicit-any]
        self: HadesEnvironment,
        action: ActType,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Takes a step in the environment.

        Args:
            action: The action to take.

        Raises:
            RuntimeError: If the game is not initialised.

        Returns:
            The observation, reward, done flag, truncated flag, and info.
        """
        # Check if the game is initialised or not
        if not self.game:
            error = "Game is not initialised"
            raise RuntimeError(error)

        # For all directions, send a key_release event to stop previous movement
        for direction in (key.W, key.A, key.S, key.D):
            self.game.game_engine.on_key_release(direction, 0)

        # Send the current action's key press event
        for direction in self._action_to_direction[action]:
            self.game.game_engine.on_key_press(direction, 0)

        # Update and render the game to reflect the new action
        self.render()

        # Check if the agent is outside the dungeon
        observations = self._get_obs()
        sprites = get_sprites_at_point(
            self.game.registry.get_component(
                self.game.player,
                KinematicComponent,
            ).get_position(),
            self.game.sprites,
        )
        if not sprites or any(
            sprite.game_object_type == GameObjectType.Wall for sprite in sprites
        ):
            return {}, 0, False, True, {}

        # Add the current position to the position history
        current_position = (
            observations["current_position"][0] // SPRITE_SIZE,
            observations["current_position"][1] // SPRITE_SIZE,
        )
        self.position_history.append(current_position)

        # Start with a neutral reward to encourage exploration
        reward = 0.5

        # Determine if the agent is near a wall or not
        # TODO: These rewards need a bit of tweaking
        min_distance_to_wall = np.min(observations["distance_to_walls"])
        if min_distance_to_wall <= SPRITE_SIZE / 2:
            # Strong penalty for touching a wall
            reward = 0.0
        elif min_distance_to_wall < SAFE_DISTANCE:
            # Penalty for being too close to the wall based on proximity to the wall
            reward -= 0.3 * (1 - (min_distance_to_wall / SAFE_DISTANCE))
        else:
            # Reward for being safely away from walls
            reward += 0.2

        # Determine if the agent is moving around or not
        if len(self.position_history) > POSITION_HISTORY_SIZE:
            # Maintain a fixed size for the history
            self.position_history.pop(0)
            if any(pos == current_position for pos in self.position_history):
                # Penalty for being stationary for too long
                reward -= 0.2
            elif (
                self.position_history.count(current_position)
                > POSITION_HISTORY_SIZE / 5
            ):
                # Penalty for oscillating on the same tile
                reward -= 0.1
            else:
                # Reward for moving around
                reward += 0.3

        # Store the action for the next step and then return the results of this step
        self.previous_action = action
        return observations, reward, False, False, {}

    def render(self: HadesEnvironment) -> None:
        """Renders the environment."""
        # Update the pyglet clock
        clock.tick()

        # Call the events for the window and render it
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event("on_draw")
        self.window.flip()

    def close(self: HadesEnvironment) -> None:
        """Closes the environment."""
        self.window.close()
