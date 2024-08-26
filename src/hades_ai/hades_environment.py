"""Manages the environment definition for the reinforcement learning agent."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Any, Final, SupportsFloat

# Pip
import numpy as np
from arcade import key
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from pyglet import app, clock

# Custom
from hades.views.game import Game
from hades_ai.capture_window import CaptureWindow
from hades_extensions.game_objects import SPRITE_SIZE, Vec2d, wall_distances
from hades_extensions.game_objects.components import KinematicComponent

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType
    from numpy.typing import NDArray

    from hades_extensions.game_objects import Space

# The rate at which to draw the game
DRAW_RATE: Final[float] = 1 / 60

# The scaling factor for incentivising movement towards the goal
GOAL_SCALING_FACTOR: Final[float] = 0.025


class HadesEnvironment(Env):
    """Represents the reinforcement learning environment for Hades.

    Attributes:
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.

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
        - distance_to_goal: The Euclidean distance to the goal.
        - direction_to_goal: The direction vector from the player to the goal.
        - previous_action: The previous action taken.

    Info Space:
        The info space is a dictionary with the following information:
        - wall_positions: The positions of the walls in the environment.
        - goal_position: The position of the goal in the environment.

    Reward Function:
        The reward function is as follows:
        - The agent is incentivized to move towards the goal based on the dot
          product of their current velocity and the direction to the goal.
        - The agent is penalised for touching a wall.
        - The agent is rewarded for reaching the goal.

    Done Function:
        The done function is as follows:
        - The agent is done if they touch a wall.
        - The agent is done if they reach the goal.
    """

    __slots__ = (
        "_action_to_direction",
        "action_space",
        "game",
        "kinematic_component",
        "observation_space",
        "previous_action",
        "space",
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
                "direction_to_goal": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "distance_to_goal": Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "distance_to_walls": Box(
                    low=0,
                    high=np.inf,
                    shape=(8,),
                    dtype=np.float32,
                ),
                "previous_action": Discrete(9),  # TODO: Is this needed?
            },
        )
        # TODO: Maybe add steering force

        # Store some variables for the environment
        self.window: CaptureWindow = CaptureWindow()
        self.game: Game | None = None
        self.kinematic_component: KinematicComponent | None = None
        self.previous_action: int = 0
        self.space: Space | None = None
        self.target_position: NDArray[float] = np.array([0.0, 0.0], dtype=np.float32)

    def _get_obs(self: HadesEnvironment) -> ObsType:
        """Returns the current observation.

        Raises:
            ValueError: If the kinematic component or space is not initialised.

        Returns:
            The current observation.
        """
        # Check if the kinematic component is initialised or not
        if not self.kinematic_component:
            error = "Kinematic component is not initialised"
            raise ValueError(error)

        # Check if the space is initialised or not
        if not self.space:
            error = "Chipmunk2D space is not initialised"
            raise ValueError(error)

        # Get the current position as a numpy array
        current_position = np.array(
            self.kinematic_component.get_position(),
            dtype=np.float32,
        )

        # Get the Euclidean distance to the goal
        distance_to_goal = np.linalg.norm(self.target_position - current_position)

        # Return the observations
        return {
            "current_position": current_position,
            "current_velocity": np.array(
                self.kinematic_component.get_velocity(),
                dtype=np.float32,
            ),
            "direction_to_goal": (
                (self.target_position - current_position) / distance_to_goal
            ),
            "distance_to_goal": np.array(
                [distance_to_goal],
                dtype=np.float32,
            ),
            "distance_to_walls": np.array(
                [
                    np.linalg.norm(
                        np.array([wall.x, wall.y], dtype=np.float32) - current_position,
                    )
                    for wall in wall_distances(
                        self.space,
                        Vec2d(*self.kinematic_component.get_position()),
                    )
                ],
                dtype=np.float32,
            ),
            "previous_action": self.previous_action,
        }

    def _get_info(self: HadesEnvironment) -> dict[str, Any]:
        """Returns the current information.

        Raises:
            ValueError: If the game is not initialised.

        Returns:
            The current information.
        """
        # Check if the game is initialised or not
        if not self.game:
            error = "Game is not initialised"
            raise ValueError(error)

        # Return information that could be useful for the agent but is not part of the
        # observation
        return {
            "wall_positions": [
                wall.position
                for wall in self.game.tile_sprites
                if wall.constructor.static
            ],
            "goal_position": np.array(self.target_position, dtype=np.float32),
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
        self.kinematic_component = self.game.registry.get_component(
            self.game.player.game_object_id,
            KinematicComponent,
        )
        self.space = self.game.registry.get_space()
        self.target_position = np.array(self.game.item_sprites[0].position)

        # Show the game view and render it so that the AI agent can interact with it
        self.window.show_view(self.game)
        self.render()

        # Return the initial observation and information
        return self._get_obs(), self._get_info()

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
            The observation, reward, done flag, info, and truncated flag.
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

        # Get observations and then save the action
        observations = self._get_obs()
        self.previous_action = action

        # Incentivize moving towards the goal based on their current velocity
        reward = (
            np.dot(observations["direction_to_goal"], observations["current_velocity"])
            * GOAL_SCALING_FACTOR
        )

        # If we're touching a wall, we're done
        done = False
        if min(observations["distance_to_walls"]) <= SPRITE_SIZE / 2:
            reward -= 5
            done = True

        # If we're at the goal, we're also done
        if observations["distance_to_goal"] <= SPRITE_SIZE / 2:
            reward += 5
            done = True

        # Return the results of this step
        return observations, reward, done, False, self._get_info()

    def render(self: HadesEnvironment) -> None:  # noqa: PLR6301
        """Renders the environment."""
        # Update the pyglet clock
        clock.tick()

        # Call the events for each window and update them
        for window in app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event("on_draw")
            window.flip()

    def close(self: HadesEnvironment) -> None:
        """Closes the environment."""
        self.window.close()


# TODO: Look at removing goal
# TODO: Introduce dungeon generation
