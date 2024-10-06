"""Manages the environment definition for the reinforcement learning agent."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Any, Final, SupportsFloat

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

# The rate at which to draw the game
DRAW_RATE: Final[float] = 1 / 60

# The safe distance to be from a wall
SAFE_DISTANCE: Final[float] = SPRITE_SIZE * 1.5

# The size of the position history
POSITION_HISTORY_SIZE: Final[int] = 50


class HadesEnvironment(Env):
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

    Done Function:
        The done function is as follows:
        - The agent is done if they are outside the dungeon or inside a wall.
    """

    __slots__ = (
        "_action_to_direction",
        "action_space",
        "game",
        "observation_space",
        "previous_action",
        "target_position",
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
        self.action_space: Discrete = Discrete(9)

        # Define the observation space
        self.observation_space = Dict(
            {
                "current_position": Box(
                    low=0,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "current_velocity": Box(
                    # This matches the MAX_VELOCITY in steering.cpp
                    low=-200,
                    high=200,
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
            ValueError: If the kinematic component or space is not initialised.

        Returns:
            The current observation.
        """
        # Check if the game is initialised or not
        if not self.game:
            error = "Game is not initialised."
            raise ValueError(error)

        # Calculate the distances to the walls
        kinematic_component = self.game.registry.get_component(
            self.game.player.game_object_id,
            KinematicComponent,
        )
        current_position = np.array(
            kinematic_component.get_position(),
            dtype=np.float32,
        )
        wall_distances = np.array(
            [
                np.linalg.norm(
                    np.array([wall.x, wall.y], dtype=np.float32) - current_position,
                )
                for wall in self.game.registry.get_system(
                    PhysicsSystem,
                ).get_wall_distances(Vec2d(*kinematic_component.get_position()))
            ],
            dtype=np.float32,
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
            "is_near_wall": float(np.any(wall_distances <= (SPRITE_SIZE / 2))),
        }

    def reset(
        self,
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

        # Show the game view and render it so that the AI agent can interact with it
        self.window.show_view(self.game)
        self.render()

        # Return the initial observation and information
        return self._get_obs(), {}

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Takes a step in the environment.

        Args:
            action: The action to take.

        Raises:
            ValueError: If the game is not initialised.

        Returns:
            The observation, reward, done flag, truncated flag, and info.
        """
        # Check if the game is initialised or not
        if not self.game:
            error = "Game is not initialised"
            raise ValueError(error)

        # For all directions, send a key_release event
        for direction in (key.W, key.A, key.S, key.D):
            self.game.on_key_release(direction, 0)

        # Get the direction from the action and send the event key
        for direction in self._action_to_direction[action]:
            self.game.on_key_press(direction, 0)

        # Update the game and render it
        self.render()

        # Get the observations and store the previous action
        observations = self._get_obs()
        self.previous_action = action

        # Check if the agent is outside the dungeon
        sprites = get_sprites_at_point(
            self.game.player.position,
            self.game.tile_sprites,
        )
        if not sprites or sprites[0].game_object_type == GameObjectType.Wall:
            return observations, 0, False, True, {}

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
            elif self.position_history.count(current_position) > 5:
                # Penalty for oscillating on the same tile
                reward -= 0.1
            else:
                # Reward for moving around
                reward += 0.3

        # Return the observations, reward, done flag, truncated flag, and info
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
