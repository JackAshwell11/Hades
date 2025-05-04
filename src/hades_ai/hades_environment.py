"""Manages the environment definition for the reinforcement learning agent."""

from __future__ import annotations

# Builtin
import time
from typing import TYPE_CHECKING, Any, Final, cast

# Pip
import numpy as np
from arcade import MOUSE_BUTTON_LEFT, key
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from pyglet import clock

# Custom
from hades.views.game import Game
from hades_ai.capture_window import CaptureWindow
from hades_extensions import GameEngine
from hades_extensions.ecs import (
    SPRITE_SIZE,
    EventType,
    GameObjectType,
    Registry,
    RegistryError,
    Vec2d,
)
from hades_extensions.ecs.components import Attack, KinematicComponent
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

# The update rate of the game
UPDATE_RATE: Final[float] = 1 / 60


class HadesEnvironment(Env):  # type:ignore[misc]
    """Represents the reinforcement learning environment for Hades.

    Attributes:
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        window: The window to capture the game view.
        game: The game view.
        game_engine: The game engine.
        registry: The registry for the game engine.
        previous_action: The previous action taken.
        position_history: The history of the player's positions.
        enemy_ids: The IDs of the enemies in the game.
        last_update_time: The time of the last headless game update.
        level: The level to play in the game environment.
        seed: The seed for the game engine.

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
        - 9: Press left mouse button.

    Observation Space:
        The observation space is a dictionary with the following observations:
        - current_position: The current position of the player.
        - current_velocity: The current velocity of the player.
        - distance_to_walls: The distances from the player to the walls in each
            direction.
        - distance_to_nearest_enemy: The distance to the nearest enemy.
        - direction_to_nearest_enemy: The compass direction to the nearest enemy.
        - time_until_attack: The time until the player can attack again.
        - previous_action: The previous action taken.
        - is_near_wall: A flag indicating if the player is near a wall or not.
        - is_near_enemy: A flag indicating if the player is near an enemy or not.

    Info Space:
        The info space is an empty dictionary.

    Reward Function:
        The reward function is as follows:
        - The agent is rewarded for staying at a safe distance from the nearest wall.
        - The agent is rewarded for moving away from enemies.
        - The agent is rewarded for attacking enemies and penalised for attacking too
            soon or not attacking at all.
        - The agent is rewarded for exploring new tiles.

    Termination Conditions:
        The episode is terminated when any of the following conditions are met:
        - The agent is outside the bounds of the dungeon.
        - The agent is inside a wall.
        - An error occurs within the ECS.
    """

    __slots__ = (
        "_action_to_key",
        "action_space",
        "enemy_ids",
        "game",
        "game_engine",
        "last_update_time",
        "level",
        "observation_space",
        "position_history",
        "previous_action",
        "registry",
        "seed",
        "window",
    )

    def __init__(self: HadesEnvironment) -> None:
        """Initialise the object."""
        super().__init__()

        # Define the action space and the mapping from actions to key presses
        self._action_to_key: dict[int, list[int]] = {
            0: [],
            1: [key.W],
            2: [key.A],
            3: [key.S],
            4: [key.D],
            5: [key.W, key.D],
            6: [key.W, key.A],
            7: [key.S, key.D],
            8: [key.S, key.A],
            9: [MOUSE_BUTTON_LEFT],
        }
        self.action_space: Discrete = Discrete(len(self._action_to_key))

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
                "time_until_attack": Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "previous_action": Discrete(9),
                "is_near_wall": Discrete(2),
                "is_near_enemy": Discrete(2),
            },
        )

        # Store some variables for the environment
        self.window: CaptureWindow | None = None
        self.game: Game | None = None
        self.game_engine: GameEngine = cast("GameEngine", None)
        self.registry: Registry = cast("Registry", None)
        self.previous_action: int = 0
        self.position_history: NDArray[NDArray[float]] = np.zeros(
            (0, 2),
            dtype=np.float32,
        )
        self.enemy_ids: set[int] = set()
        self.last_update_time: float = time.time()
        self.level: int = 0
        self.seed: int | None = None

    @property
    def show_window(self: HadesEnvironment) -> bool:
        """Whether to show the Pyglet window or not.

        Returns:
            Whether to show the Pyglet window or not.
        """
        return self.window is not None

    @show_window.setter
    def show_window(self: HadesEnvironment, value: bool) -> None:
        """Set whether to show the Pyglet window or not.

        Args:
            value: Whether to show the Pyglet window or not.
        """
        if value and not self.window:
            self.window = CaptureWindow()
            self.game = Game()
            self.window.show_view(self.game)
        elif not value:
            self.window = self.game = None

    def _get_enemy_positions(self: HadesEnvironment) -> NDArray[NDArray[float]]:
        """Returns the positions of the enemies.

        Returns:
            The positions of the enemies.
        """
        return np.array(
            [
                self.registry.get_component(
                    enemy_id,
                    KinematicComponent,
                ).get_position()
                for enemy_id in self.enemy_ids
            ],
            dtype=np.float32,
        )

    def _get_obs(self: HadesEnvironment) -> ObsType:
        """Returns the current observation.

        Returns:
            The current observation.
        """
        # Calculate the distances to the walls
        kinematic_component = self.registry.get_component(
            self.game_engine.player_id,
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
                    for wall in self.registry.get_system(
                        PhysicsSystem,
                    ).get_wall_distances(Vec2d(*current_position))
                ],
            )
            - current_position,
            axis=1,
        )

        # Get the enemy positions and calculate the distance to the nearest enemy
        enemy_positions = self._get_enemy_positions()
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
            "distance_to_nearest_enemy": distance_to_nearest_enemy,
            "direction_to_nearest_enemy": (
                (enemy_positions[nearest_enemy_index] - current_position)
                / distance_to_nearest_enemy
            ),
            "time_until_attack": np.array(
                self.registry.get_component(
                    self.game_engine.player_id,
                    Attack,
                ).time_until_ranged_attack,
                dtype=np.float32,
            ),
            "previous_action": self.previous_action,
            "is_near_wall": np.float32(np.any(wall_distances <= WALL_SAFE_DISTANCE)),
            "is_near_enemy": np.float32(
                distance_to_nearest_enemy <= ENEMY_SAFE_DISTANCE,
            ),
        }

    def _update_player_rotation(self: HadesEnvironment) -> None:
        """Updates the player's rotation to face the nearest enemy."""
        # Create a vector towards the nearest enemy and set the player's rotation to it
        player = self.registry.get_component(
            self.game_engine.player_id,
            KinematicComponent,
        )
        player_position = player.get_position()
        enemy_positions = self._get_enemy_positions()
        distances = np.linalg.norm(enemy_positions - player_position, axis=1)
        dx, dy = enemy_positions[np.argmin(distances)] - player_position
        player.set_rotation(np.arctan2(dy, dx))

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

        # Reset the environment and store the required variables
        self.previous_action = 0
        if self.game:
            self.game.setup(self.level, self.seed)
            self.game_engine = self.game.game_engine
        else:
            self.game_engine = GameEngine(self.level, self.seed)
            self.game_engine.create_game_objects()
        self.registry = self.game_engine.get_registry()
        self.registry.add_callback(
            EventType.GameObjectCreation,
            self.on_game_object_creation,
        )
        self.registry.add_callback(
            EventType.GameObjectDeath,
            self.on_game_object_death,
        )
        self.position_history = np.zeros((0, 2), dtype=np.float32)
        self.enemy_ids.clear()

        # Generate the maximum number of enemies and show the new game view
        _, _, enemy_limit = self.game_engine.level_constants
        for _ in range(enemy_limit):
            self.game_engine.generate_enemy(0)

        # Reset the environment if anything has gone wrong
        if len(self.enemy_ids) < enemy_limit:
            return self.reset(seed=seed, options=options)

        # Return the initial observation and information
        return self._get_obs(), {}

    def step(  # type: ignore[explicit-any]
        self: HadesEnvironment,
        action: ActType,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Takes a step in the environment.

        Args:
            action: The action to take.

        Returns:
            The observation, reward, done flag, truncated flag, and info.
        """
        # For all directions, send a key_release event to stop previous movement
        for direction in (key.W, key.A, key.S, key.D):
            self.game_engine.on_key_release(direction, 0)

        # Face the nearest enemy and send the current action's press event
        self._update_player_rotation()
        attack_successful = False
        for action_key in self._action_to_key[action]:
            if action_key == MOUSE_BUTTON_LEFT:
                attack_successful = self.game_engine.on_mouse_press(
                    0,
                    0,
                    action_key,
                    0,
                )
            else:
                self.game_engine.on_key_press(action_key, 0)

        # Update and render the game to reflect the new action
        try:
            self.render()
        except RegistryError:
            return {}, 0, False, True, {}

        # Calculate the wall avoidance reward component for staying away from walls
        observations = self._get_obs()
        min_distance_to_wall = np.min(observations["distance_to_walls"])
        wall_avoidance_reward = (
            1 - np.exp(-min_distance_to_wall / (WALL_SAFE_DISTANCE * 0.5))
            if min_distance_to_wall > (SPRITE_SIZE / 2)
            else -1
        )

        # Calculate the empowerment reward component for moving away from enemies
        empowerment_reward = (
            1
            - np.maximum(
                0,
                ENEMY_SAFE_DISTANCE - observations["distance_to_nearest_enemy"],
            )
            / ENEMY_SAFE_DISTANCE
        )

        # Calculate the attack reward component for attacking enemies
        attack_reward = 0.5
        if observations["time_until_attack"] == 0:
            attack_reward = 1.0 if attack_successful else 0.0
        elif self._action_to_key[action] == [MOUSE_BUTTON_LEFT]:
            attack_reward = 0.0

        # Calculate the exploration reward component for new tiles
        agent_grid_position = observations["current_position"] // SPRITE_SIZE
        exploration_reward = float(
            not np.any(np.all(agent_grid_position == self.position_history, axis=1)),
        )
        self.position_history = np.vstack((self.position_history, agent_grid_position))[
            -POSITION_HISTORY_SIZE:
        ]

        # Store the action for the next step and then return the results of this step
        self.previous_action = action
        return (
            observations,
            wall_avoidance_reward
            + empowerment_reward
            + attack_reward
            + exploration_reward,
            False,
            False,
            {},
        )

    def render(self: HadesEnvironment) -> None:
        """Renders the environment."""
        if self.window:
            # Update the pyglet clock
            clock.tick()

            # Call the events for the window and render it
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.dispatch_event("on_draw")
            self.window.flip()
        else:
            # Use the elapsed time to update the game to prevent it from
            # running too fast
            self.game_engine.generate_enemy(0)
            current_time = time.time()
            if (current_time - self.last_update_time) >= UPDATE_RATE:
                self.registry.update(UPDATE_RATE)
                self.last_update_time = current_time

    def close(self: HadesEnvironment) -> None:
        """Closes the environment."""
        if self.window:
            self.window = self.game = None

    def on_game_object_creation(self: HadesEnvironment, game_object_id: int) -> None:
        """Add a game object to the game.

        Args:
            game_object_id: The ID of the game object to add.
        """
        if self.registry.get_game_object_type(game_object_id) == GameObjectType.Enemy:
            self.enemy_ids.add(game_object_id)

    def on_game_object_death(self: HadesEnvironment, game_object_id: int) -> None:
        """Remove a game object from the game.

        Args:
            game_object_id: The ID of the game object to remove.

        Raises:
            RegistryError: If the player has died.
        """
        if self.registry.get_game_object_type(game_object_id) == GameObjectType.Enemy:
            self.enemy_ids.remove(game_object_id)
        elif game_object_id == self.game_engine.player_id:
            error = "Player has died."
            raise RegistryError(error)
