"""Manages the environment definition for the reinforcement learning agent."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Any, SupportsFloat

# Pip
import numpy as np
from arcade import key
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from pyglet import app, clock

# Custom
from hades.views.game import Game
from hades_ai.capture_window import CaptureWindow
from hades_extensions.ecs.components import KinematicComponent

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType


class HadesEnvironment(Env):
    """Represents the reinforcement learning environment for Hades.

    Attributes:
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
    """

    __slots__ = (
        "_action_to_direction",
        "action_space",
        "game",
        "kinematic_component",
        "observation_space",
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
            },
        )

        # Store some variables for the environment
        self.window: CaptureWindow = CaptureWindow()
        self.game: Game | None = None
        self.kinematic_component: KinematicComponent | None = None

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

        # Return the observations
        return {
            "current_position": np.array(
                self.kinematic_component.get_position(),
                dtype=np.float32,
            ),
            "current_velocity": np.array(
                self.kinematic_component.get_velocity(),
                dtype=np.float32,
            ),
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

        # Return information that could be useful for the agent but are not part of the
        # observation
        return {}

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
        self.game = Game(0)
        self.kinematic_component = self.game.registry.get_component(
            self.game.player,
            KinematicComponent,
        )

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

        # Return the results of this step
        return self._get_obs(), 1, True, False, self._get_info()

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
