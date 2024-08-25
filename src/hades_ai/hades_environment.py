"""Manages the environment definition for the reinforcement learning agent."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Any, Final

# Pip
import numpy as np
from arcade import key
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from pyglet import app, clock

# Custom
from hades.views.game import Game
from hades.window import HadesWindow
from hades_extensions.ecs.components import KinematicComponent

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType

__all__ = ("HadesEnvironment",)

# The maximum velocity of the player
MAX_VELOCITY: Final[float] = 600


class HadesEnvironment(Env):  # type:ignore[misc]
    """Represents the reinforcement learning environment for Hades.

    Attributes:
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        window: The window to capture the game view.
        game: The game view.
        kinematic_component: The kinematic component of the player.

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

    Info Space:
        The info space is an empty dictionary.

    Reward Function:
        The reward function is as follows:
        - Always returns a reward of 1.

    Termination Conditions:
        The episode is terminated when any of the following conditions are met:
        - Always returns True.
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
            },
        )

        # Store some variables for the environment
        self.window: HadesWindow = HadesWindow()
        self.game: Game | None = None
        self.kinematic_component: KinematicComponent | None = None

    def _get_obs(self: HadesEnvironment) -> ObsType:
        """Returns the current observation.

        Raises:
            RuntimeError: If the kinematic component is not initialised.

        Returns:
            The current observation.
        """
        # Check if the kinematic component is initialised or not
        if not self.kinematic_component:
            error = "Kinematic component is not initialised"
            raise RuntimeError(error)

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
        self.game = Game(0)
        self.kinematic_component = self.game.registry.get_component(
            self.game.player,
            KinematicComponent,
        )

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

        # Return the results of this step
        return self._get_obs(), 1, True, False, {}

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
