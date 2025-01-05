"""Manages the environment definition for the reinforcement learning agent."""

from __future__ import annotations

# Builtin
import time
from typing import TYPE_CHECKING, Any, Final, SupportsFloat

# Pip
import numpy as np
from arcade import key
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from pyglet import app

# Custom
from hades.views.game import Game
from hades_ai.capture_window import CaptureWindow
from hades_extensions.ecs import (
    SPRITE_SIZE,
    EventType,
    GameObjectType,
    RegistryError,
    Vec2d,
)
from hades_extensions.ecs.components import KinematicComponent
from hades_extensions.ecs.systems import PhysicsSystem

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType
    from numpy.typing import NDArray

__all__ = ("HadesEnvironment",)

# The maximum velocity of the player
MAX_VELOCITY: Final[float] = 600

# The safe distance to be from a wall
WALL_SAFE_DISTANCE: Final[float] = SPRITE_SIZE * 1.5

# The safe distance to be from an enemy
ENEMY_SAFE_DISTANCE: Final[float] = SPRITE_SIZE * 3

# The size of the position history
POSITION_HISTORY_SIZE: Final[int] = 25

# The grid tile distance to be from an enemy
ENEMY_SAFE_GRID_DISTANCE: Final[int] = 5


def app_exit() -> None:
    """Stop the game window from closing and instead create a new episode.

    Raises:
        RegistryError: If the player has died or reached the goal.
    """
    # TODO: This will need to be changed when goal behaviour is implemented
    raise RegistryError


class HadesEnvironment(Env):
    """Represents the reinforcement learning environment for Hades.

    Attributes:
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        window: The window to capture the game view.
        game: The game view.
        previous_action: The previous action taken.
        position_history: The history of the player's positions.
        enemy_ids: The IDs of the enemies in the game.
        last_update_time: The time of the last update.

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
        - distance_to_nearest_enemy: The distance to the nearest enemy.
        - direction_to_nearest_enemy: The compass direction to the nearest enemy.
        - is_near_enemy: A flag indicating if the player is near an enemy or not.

    Info Space:
        The info space is an empty dictionary.

    Reward Function:
        The reward function is as follows:
        - Start with an empowerment reward for moving away from enemies.
        - The agent is rewarded for staying at a safe distance from the nearest enemy.
        - The agent is rewarded for staying at a safe distance from the nearest wall.
        - The agent is rewarded for exploring new tiles.

    Finish Condition:
        The episode is finished if:
        - The agent is outside the bounds of the dungeon.
        - The agent is inside a wall.
        - An error occurs within the ECS.
    """

    __slots__ = (
        "_action_to_direction",
        "action_space",
        "enemy_ids",
        "game",
        "last_update_time",
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
        self.observation_space = Dict(
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
                "distance_to_nearest_enemy": Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "direction_to_nearest_enemy": Box(
                    low=-1,
                    high=1,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "is_near_enemy": Discrete(2),
            },
        )

        # Store some variables for the environment
        self.window: CaptureWindow = CaptureWindow()
        self.game: Game | None = None
        self.previous_action: int = 0
        self.position_history: NDArray[NDArray[float]] = np.zeros(
            (0, 2),
            dtype=np.float32,
        )
        self.enemy_ids: set[int] = set()
        self.last_update_time: float = 0
        app.exit = app_exit

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
                    (wall.x, wall.y)
                    for wall in self.game.registry.get_system(
                        PhysicsSystem,
                    ).get_wall_distances(Vec2d(*current_position))
                ],
            )
            - current_position,
            axis=1,
        )

        # Get the enemy positions and calculate the distance to the nearest enemy
        enemy_positions = np.zeros((len(self.enemy_ids), 2), dtype=np.float32)
        for i, enemy_id in enumerate(self.enemy_ids):
            enemy_positions[i] = self.game.registry.get_component(
                enemy_id,
                KinematicComponent,
            ).get_position()
        distances_to_enemies = np.linalg.norm(
            enemy_positions - current_position,
            axis=1,
        )
        nearest_enemy_index = np.argmin(distances_to_enemies)
        distance_to_nearest_enemy = distances_to_enemies[nearest_enemy_index]

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
            "distance_to_nearest_enemy": distance_to_nearest_enemy,
            "direction_to_nearest_enemy": (
                enemy_positions[nearest_enemy_index] - current_position
            )
            / distance_to_nearest_enemy,
            "is_near_enemy": float(np.any(distances_to_enemies <= ENEMY_SAFE_DISTANCE)),
        }

    def _calculate_empowerment(
        self: HadesEnvironment,
        current_position: np.ndarray,
    ) -> float:
        """Encourage movement towards tiles furthest away from enemies.

        Args:
            current_position: The current position of the player.

        Raises:
            RuntimeError: If the game is not initialised.

        Returns:
            The empowerment value.
        """
        # Check if the game is initialised or not
        if not self.game:
            error = "Game is not initialised."
            raise RuntimeError(error)

        # Calculate grid distances to all enemies from the player's grid position
        distances_to_enemies = [
            np.linalg.norm(
                current_position
                - np.array(
                    self.game.registry.get_component(
                        enemy_id,
                        KinematicComponent,
                    ).get_position(),
                    dtype=np.float32,
                )
                // SPRITE_SIZE,
            )
            for enemy_id in self.enemy_ids
        ]

        # Calculate empowerment based on the inverse distance to enemies
        return (
            1
            - max(
                0,
                ENEMY_SAFE_GRID_DISTANCE
                - min(distances_to_enemies, default=ENEMY_SAFE_GRID_DISTANCE),
            )
            / ENEMY_SAFE_GRID_DISTANCE
        )

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
        self.game.registry.add_callback(
            EventType.GameObjectCreation,
            self.on_game_object_creation,
        )
        self.position_history = np.zeros((0, 2), dtype=np.float32)
        self.enemy_ids.clear()

        # Generate the maximum number of enemies, and reset the environment if not
        # enough are generated
        enemy_limit = self.game.game_engine.level_constants[2]
        for _ in range(enemy_limit):
            self.game.game_engine.generate_enemy(0)
        if len(self.enemy_ids) < enemy_limit:
            return self.reset(seed=seed, options=options)

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
            self.game.on_key_release(direction, 0)

        # Send the current action's key press event
        for direction in self._action_to_direction[action]:
            self.game.on_key_press(direction, 0)

        # Update and render the game to reflect the new action then check if the
        # episode should finish
        try:
            self.render()
        except RegistryError:
            return {}, 0, True, True, {}

        # Check if the agent is outside the dungeon
        observations = self._get_obs()
        pos = observations["current_position"]
        width, height, _ = self.game.game_engine.level_constants
        if (
            pos[0] < 0
            or pos[1] < 0
            or pos[0] > (width * SPRITE_SIZE)
            or pos[1] > (height * SPRITE_SIZE)
        ):
            return observations, 0, True, True, {}

        # Calculate the wall avoidance reward component for staying away from walls
        # TODO: Properly normalise this
        min_distance_to_wall = np.min(observations["distance_to_walls"])
        wall_avoidance_reward = (
            1 - np.exp(-min_distance_to_wall / WALL_SAFE_DISTANCE)
            if min_distance_to_wall > (SPRITE_SIZE / 2)
            else -5
        )

        # Calculate the empowerment reward component for moving away from enemies
        agent_grid_position = np.array(observations["current_position"] // SPRITE_SIZE)
        empowerment_reward = self._calculate_empowerment(agent_grid_position)

        # Calculate the enemy avoidance reward component for staying away from enemies
        # TODO: Properly normalise this
        distance_to_near_enemy = observations["distance_to_nearest_enemy"]
        enemy_avoidance_reward = (
            1 - (distance_to_near_enemy / ENEMY_SAFE_DISTANCE)
            if distance_to_near_enemy < ENEMY_SAFE_DISTANCE
            else 0
        )

        # Calculate the exploration reward component for new tiles
        exploration_reward = 0
        if not np.any(
            np.all(self.position_history == agent_grid_position, axis=1),
        ):
            exploration_reward = 1
            self.position_history = np.vstack(
                (self.position_history, agent_grid_position),
            )
        if len(self.position_history) > POSITION_HISTORY_SIZE:
            self.position_history = self.position_history[1:]

        # Store the action for the next step and then return the results of this step
        self.previous_action = action
        return (
            observations,
            wall_avoidance_reward
            + empowerment_reward
            + enemy_avoidance_reward
            + exploration_reward,
            False,
            False,
            {},
        )

    def render(self: HadesEnvironment) -> None:
        """Renders the environment.

        Raises:
            RuntimeError: If the game is not initialised.
        """
        # Check if the game is initialised or not
        if not self.game:
            error = "Game is not initialised."
            raise RuntimeError(error)

        # Get the delta time and do one step of the game
        current_time = time.time()
        self.game.on_update(current_time - self.last_update_time)
        self.game.on_draw()
        self.window.on_update(current_time - self.last_update_time)
        self.window.flip()
        self.last_update_time = current_time

    def close(self: HadesEnvironment) -> None:
        """Closes the environment."""
        self.window.close()

    def on_game_object_creation(self: HadesEnvironment, game_object_id: int) -> None:
        """Add a game object to the game.

        Args:
            game_object_id: The ID of the game object to add.

        Raises:
            RuntimeError: If the game is not initialised.
        """
        # Check if the game is initialised or not
        if not self.game:
            error = "Game is not initialised."
            raise RuntimeError(error)

        # Add the game object to the collection if it represents an enemy
        if (
            self.game.registry.get_game_object_type(game_object_id)
            == GameObjectType.Enemy
        ):
            self.enemy_ids.add(game_object_id)
