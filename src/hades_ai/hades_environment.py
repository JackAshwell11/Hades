"""Manages the environment definition for the reinforcement learning agent."""

from __future__ import annotations

# Builtin
import time
from typing import TYPE_CHECKING, Any, Final, SupportsFloat

# Pip
import numpy as np
from arcade import get_sprites_at_point, key
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete

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
from hades_extensions.ecs.components import KinematicComponent, PythonSprite
from hades_extensions.ecs.systems import PhysicsSystem

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType

# The safe distance to be from a wall
SAFE_DISTANCE: Final[float] = SPRITE_SIZE * 1.5

# The size of the position history
POSITION_HISTORY_SIZE: Final[int] = 25


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
        "last_update_time",
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
                "distance_to_enemies": Box(
                    low=0,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "is_near_enemy": Discrete(2),
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
                # "empowerment": Box(
                #     low=0,
                #     high=1,
                #     shape=(1,),
                #     dtype=np.float32,
                # ),
            },
        )

        # Store some variables for the environment
        self.window: CaptureWindow = CaptureWindow()
        self.window.center_window()
        self.game: Game | None = None
        self.previous_action: int = 0
        self.position_history: list[tuple[int, int]] = []
        self.enemy_ids: list[int] = []
        self.empowerment_matrix = None
        self.finished: bool = False
        self.last_update_time = 0

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

        try:
            kinematic_component = self.game.registry.get_component(
                self.game.player,
                KinematicComponent,
            )
        except RegistryError:
            return {}
        current_position = np.array(
            kinematic_component.get_position(),
            dtype=np.float32,
        )

        wall_positions = np.array(
            [
                [wall.x, wall.y]
                for wall in self.game.registry.get_system(
                    PhysicsSystem,
                ).get_wall_distances(Vec2d(*kinematic_component.get_position()))
            ],
            dtype=np.float32,
        )
        wall_distances = np.linalg.norm(wall_positions - current_position, axis=1)

        enemy_positions = []
        distance_to_enemies = []
        for enemy_id in self.enemy_ids:
            enemy_kinematic_component = self.game.registry.get_component(
                enemy_id,
                KinematicComponent,
            )
            enemy_position = np.array(
                enemy_kinematic_component.get_position(),
                dtype=np.float32,
            )
            distance_to_enemies.append(
                np.linalg.norm(current_position - enemy_position),
            )
            enemy_positions.append(enemy_position)
        enemy_positions = np.array(enemy_positions, dtype=np.float32)
        distance_to_enemies = np.array(distance_to_enemies, dtype=np.float32)

        if not enemy_positions.size:
            direction_to_nearest_enemy = np.zeros(2, dtype=np.float32)
        else:
            direction_to_nearest_enemy = (
                current_position - enemy_positions[np.argmin(distance_to_enemies)]
            )

        # Ensure distance_to_enemies has a fixed size
        if len(distance_to_enemies) < 3:
            distance_to_enemies = np.pad(
                distance_to_enemies,
                (0, 3 - len(distance_to_enemies)),
                "constant",
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
            "distance_to_enemies": distance_to_enemies,
            "is_near_enemy": float(np.any(distance_to_enemies <= (SAFE_DISTANCE * 3))),
            "distance_to_nearest_enemy": np.min(distance_to_enemies),
            "direction_to_nearest_enemy": direction_to_nearest_enemy,
        }

    def _calculate_empowerment(self, current_position: np.ndarray) -> float:
        """Calculate empowerment based on inverse distance to enemies."""
        if not self.enemy_ids:
            return 1.0  # Maximum empowerment if there are no enemies

        # Calculate grid distances to all enemies from the player's grid position
        distances_to_enemies = [
            np.linalg.norm(
                current_position
                - np.array(
                    (
                        int(
                            self.game.registry.get_component(
                                enemy_id,
                                KinematicComponent,
                            ).get_position()[0]
                            // SPRITE_SIZE,
                        ),
                        int(
                            self.game.registry.get_component(
                                enemy_id,
                                KinematicComponent,
                            ).get_position()[1]
                            // SPRITE_SIZE,
                        ),
                    ),
                ),
            )
            for enemy_id in self.enemy_ids
        ]

        # Define a threshold grid distance for high-risk zones
        ENEMY_SAFE_GRID_DISTANCE = 3  # In grid cells, adjustable for higher avoidance

        # Calculate empowerment based on grid distance: the further from enemies, the higher the empowerment
        return (
            1
            - max(
                0,
                min(
                    ENEMY_SAFE_GRID_DISTANCE - min(distances_to_enemies),
                    ENEMY_SAFE_GRID_DISTANCE,
                ),
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
        self.finished = False
        self.game.registry.add_callback(
            EventType.GameObjectDeath,
            self.on_game_object_death,
        )
        self.position_history.clear()

        # Show the game view and render it so that the AI agent can interact with it
        self.window.show_view(self.game)
        self.render()

        # Get the enemy id
        self.enemy_ids = [
            entity.game_object_id
            for entity in self.game.entity_sprites
            if entity.game_object_type == GameObjectType.Enemy
        ]

        self.empowerment_matrix = np.zeros((20, 30), dtype=np.float32)

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

        # For all directions, send a key_release event to stop previous movement
        for direction in (key.W, key.A, key.S, key.D):
            self.game.on_key_release(direction, 0)

        # Send the current action's key press event
        for direction in self._action_to_direction[action]:
            self.game.on_key_press(direction, 0)

        # Update and render the game to reflect the new action
        if self.finished:
            print("finished early")
            return {}, 0, True, True, {}
        self.render()

        # Get current observation
        observations = self._get_obs()
        if observations == {}:
            print("no obs")
            return {}, -1, True, True, {}

        # Check if the agent is outside the dungeon
        sprites = get_sprites_at_point(
            self.game.registry.get_component(
                self.game.player,
                KinematicComponent,
            ).get_position(),
            self.game.tile_sprites,
        )
        if not sprites or sprites[0].game_object_type == GameObjectType.Wall:
            print("outside dungeon")
            return observations, -1, False, True, {}

        # Get agent's current grid position
        agent_grid_position = (
            int(observations["current_position"][0] // SPRITE_SIZE),
            int(observations["current_position"][1] // SPRITE_SIZE),
        )

        # Empowerment reward: incentivising movement away from enemies
        reward = self._calculate_empowerment(agent_grid_position)

        # Penalize getting close to enemies
        MAX_PENALTY_DISTANCE = SPRITE_SIZE * 3
        for distance in observations["distance_to_enemies"]:
            if distance < MAX_PENALTY_DISTANCE:
                normalized_penalty = 1 - (distance / MAX_PENALTY_DISTANCE)
                reward -= normalized_penalty * 0.5

        # Wall avoidance reward
        min_distance_to_wall = np.min(observations["distance_to_walls"])
        if min_distance_to_wall <= (SPRITE_SIZE / 2):
            reward -= 5
        else:
            reward += 1 - np.exp(-min_distance_to_wall / SAFE_DISTANCE)

        # Exploration bonus for new grid position
        if agent_grid_position not in self.position_history:
            reward += 1.0
            self.position_history.append(agent_grid_position)
        if len(self.position_history) > POSITION_HISTORY_SIZE:
            self.position_history.pop(0)

        self.previous_action = action

        # TODO: Would like to have reward be very small (-2 to 2 for combined and -1 to
        #  1 for individual)

        # Return the observation, reward, done, truncated, and info
        return observations, reward, False, False, {}

    def render(self: HadesEnvironment) -> None:
        """Renders the environment."""
        current_time = time.time()
        try:
            self.game.on_update(current_time - self.last_update_time)
        except RegistryError:
            self.finished = True
            return
        self.game.on_draw()
        self.window.on_update(current_time - self.last_update_time)
        self.window.flip()
        self.last_update_time = current_time

    def close(self: HadesEnvironment) -> None:
        """Closes the environment."""
        self.window.close()

    def on_game_object_death(self: HadesEnvironment, game_object_id: int) -> None:
        """Remove a game object from the game.

        Args:
            game_object_id: The ID of the game object to remove.
        """
        self.game.game_ui.on_game_object_death(game_object_id)
        game_object = self.game.registry.get_component(
            game_object_id,
            PythonSprite,
        ).sprite
        game_object.remove_from_sprite_lists()
        if game_object_id == self.game.player:
            self.finished = True
            print("finished true")
            return
