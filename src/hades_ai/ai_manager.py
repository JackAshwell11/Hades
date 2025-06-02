"""Manages the reinforcement learning training for the Hades AI agent."""

from __future__ import annotations

# Builtin
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

# Pip
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

# Custom
from hades_ai.dqn import BATCH_SIZE, DQNAgent, Transition
from hades_ai.hades_environment import HadesEnvironment
from hades_extensions.ecs.components import Armour, Health

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymnasium.core import ObsType

__all__ = ()


# The number of episodes to train the agent for (larger values can provide better
# results but require more time)
EPISODE_COUNT: Final[int] = 2000

# The number of episodes to average the rewards over for the graph (larger values can
# show a smoother graph but may hide important details)
GRAPH_MOVING_AVERAGE: Final[int] = 50

# The maximum number of steps to take in each episode before stopping (larger values can
# give the agent more time to learn but may take longer to train)
MAX_STEP_COUNT: Final[int] = 2000

# The interval at which to save the gameplay and graphs (larger values can reduce the
# number of saves but may miss important details)
SAVE_INTERVAL: Final[int] = 50

# The number of episodes to evaluate the agent for (larger values can provide more
# accurate results but require more time)
EVALUATE_EPISODES: Final[int] = 1000

# The name of the model file that will be saved
MODEL_NAME: Final[str] = "model.pth"

# Get the path to the output directory
OUTPUT_DIR: Final[Path] = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Create the environment for the AI agent
ENV: Final[HadesEnvironment] = HadesEnvironment()

# Create the agent for the AI
AGENT: Final[DQNAgent] = DQNAgent(ENV.observation_space, ENV.action_space)


class BuildNamespace(Namespace):
    """Allows typing of an argparse namespace for the CLI."""

    window: bool
    level: int | None
    seed: int | None
    check: bool
    run: bool
    train: bool
    evaluate: bool


@dataclass
class EpisodeResults:
    """Stores the results of an episode.

    Attributes:
        reward: The total reward for the episode.
        loss: The total loss for the episode.
        steps: The total number of steps taken in the episode.
        survived: Whether the agent survived the episode.
        wall_distances: The average distance to walls for each step.
        enemy_distances: The distance to the nearest enemy for each step.
        healths: The health values for each step.
    """

    reward: float = 0.0
    loss: float = 0.0
    steps: int = 0
    survived: bool = True
    wall_distances: list[np.float32] = field(default_factory=list)
    enemy_distances: list[np.float32] = field(default_factory=list)
    healths: list[float] = field(default_factory=list)

    @property
    def average_reward(self: EpisodeResults) -> float:
        """Get the average reward per step.

        Returns:
            The average reward per step.
        """
        return self.reward / self.steps if self.steps > 0 else 0

    @property
    def average_loss(self: EpisodeResults) -> float:
        """Get the average loss per step.

        Returns:
            The average loss per step.
        """
        return self.loss / self.steps if self.steps > 0 else 0


def plot_metric(
    metric: Sequence[float],
    labels: tuple[str, str],
    save_dir: Path | None = OUTPUT_DIR,
    plot_num: int | None = None,
) -> None:
    """Plot a metric on a graph.

    Args:
        metric: The metric to plot.
        labels: The title and y-axis label for the graph.
        save_dir: The directory to save the graph to.
        plot_num: The number of the plot to update, otherwise, a new one will be
        created.
    """
    # Give the figure a number and title so that we can update it
    plt.figure(plot_num)
    plt.clf()
    plt.title(labels[0])

    # Label the x and y axes and plot the metric
    plt.xlabel("Episode")
    plt.ylabel(labels[1])
    plt.plot(metric, label=labels[1])

    # Plot the moving average of the metric
    moving_avg = [
        np.mean(metric[np.maximum(0, i - GRAPH_MOVING_AVERAGE + 1) : i + 1])
        for i in range(len(metric))
    ]
    plt.plot(moving_avg, label=f"Average {labels[1].lower()}")

    # Add a legend to the graph after plotting the metric (we can't add it before as
    # we need to know the label)
    plt.legend()

    # Save the graph if possible
    if save_dir:
        plt.savefig(save_dir / f"{labels[0].lower()}.svg")


def concat_observation(obs: ObsType) -> torch.Tensor:
    """Concatenate the observation into a single tensor.

    Args:
        obs: The observation.

    Returns:
        The concatenated observation tensor.
    """
    # Concatenate all parts of the observation into a single array
    arr = np.array([])
    for key in obs:
        arr = np.append(arr, obs[key])

    # Return the array as a tensor with the correct shape and device
    return torch.tensor(
        arr,
        dtype=torch.float32,
        device=AGENT.device,
    ).unsqueeze(0)


def process_episode(*, train: bool = True) -> EpisodeResults:
    """Process an episode of the environment.

    Args:
        train: Whether to train the agent or not.

    Returns:
        The results of the episode.
    """
    # Reset the environment
    state, _ = ENV.reset()

    # Ensure the state tensor includes all parts of the observation
    state = concat_observation(state)

    # Run the episode and get the results
    results = EpisodeResults()
    for _ in range(MAX_STEP_COUNT * (1 if train else 10)):
        # Select an action based on whether we're training or not then perform it and
        # get the next state
        action = AGENT.select_action(state, train=train)
        observations, reward, done, truncated, _ = ENV.step(action.item())
        results.reward += reward
        results.steps += 1
        results.survived = not truncated
        finish = truncated or done
        next_state = concat_observation(observations) if not finish else None

        # Collect data about the step
        if observations:
            results.wall_distances.append(np.mean(observations["distance_to_walls"]))
            results.enemy_distances.append(observations["distance_to_nearest_enemy"])
            results.healths.append(
                ENV.registry.get_component(
                    ENV.game_engine.player_id,
                    Health,
                ).get_value()
                + ENV.registry.get_component(
                    ENV.game_engine.player_id,
                    Armour,
                ).get_value(),
            )

        # Store the transition in memory
        if train:
            AGENT.memory.append(
                Transition(
                    state,
                    action,
                    next_state,
                    torch.tensor([reward], device=AGENT.device),
                ),
            )

        # Move to the next state
        state = next_state

        # Optimise the model if we have enough transitions
        if train and len(AGENT.memory) >= BATCH_SIZE:
            results.loss += AGENT.optimise_model()

        # If we're done, stop the current episode
        if finish:
            break
    return results


def train_dqn() -> None:
    """Train the DQN agent in the environment."""
    # Loop over the episodes
    episode_rewards = []
    episode_losses = []
    for episode in range(EPISODE_COUNT):
        # Enable saving if possible
        episode_dir = OUTPUT_DIR / f"{episode + 1}"
        if do_save := episode % SAVE_INTERVAL == 0 or episode == EPISODE_COUNT - 1:
            episode_dir.mkdir(exist_ok=True)
            if ENV.window:
                ENV.window.make_writer(episode_dir / f"episode_{episode + 1}.mp4")

        # Run the episode and print the episode results
        results = process_episode()
        print(  # noqa: T201
            f"Finished episode {episode + 1} (saving: {do_save}), after {results.steps}"
            f" steps. Average reward: {results.average_reward}, average loss:"
            f" {results.average_loss}",
        )

        # Log the rewards and losses for this episode
        episode_rewards.append(results.average_reward)
        episode_losses.append(results.average_loss)

        # Plot the graphs for the episode
        save_dir = episode_dir if do_save else None
        plot_metric(episode_rewards, ("Episode Rewards", "Reward"), save_dir, 0)
        plot_metric(episode_losses, ("Episode Losses", "Loss"), save_dir, 1)
        if ENV.show_window:
            plt.pause(0.01)

        # Save the model and video if possible
        if do_save:
            torch.save(AGENT.policy_net.state_dict(), episode_dir / MODEL_NAME)
            if ENV.window:
                ENV.window.save_video()

        # Update the target network after the episode
        AGENT.update_target_network()

    # Save the final model
    torch.save(AGENT.policy_net.state_dict(), OUTPUT_DIR / MODEL_NAME)


def run_dqn(*, evaluate: bool = False) -> None:
    """Run the DQN agent in the environment.

    Args:
        evaluate: Whether to evaluate the agent or not.
    """
    # Find the most recent model
    path = (
        max(
            (d for d in OUTPUT_DIR.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            default=OUTPUT_DIR,
        )
        / MODEL_NAME
    )
    if not path.exists():
        print("No model found, please train the model first")  # noqa: T201
        return

    # Initialise the collections for evaluation
    rewards = []
    survivals = []
    min_wall_distances = []
    average_wall_distances = []
    min_enemy_distances = []
    average_enemy_distances = []
    healths = []

    # Load the trained model then run the agent for the specified number of episodes
    AGENT.policy_net.load_state_dict(torch.load(path, weights_only=True))
    for episode in range(EVALUATE_EPISODES if evaluate else 1):
        if evaluate:
            print(  # noqa: T201
                f"Evaluating episode {episode + 1} of {EVALUATE_EPISODES}",
            )
        results = process_episode(train=False)
        rewards.append(results.average_reward)
        survivals.append(results.survived)
        min_wall_distances.append(np.min(results.wall_distances))
        average_wall_distances.append(float(np.mean(results.wall_distances)))
        min_enemy_distances.append(np.min(results.enemy_distances))
        average_enemy_distances.append(float(np.mean(results.enemy_distances)))
        healths.append(float(np.mean(results.healths)))
        if not evaluate:
            print("Average reward:", results.average_reward)  # noqa: T201

    # Display the results of the evaluation
    if evaluate:
        print(f"Average Reward: {np.mean(rewards)}")  # noqa: T201
        print(f"Survival Rate: {np.mean(survivals) * 100}%")  # noqa: T201
        print(f"Minimum Distance to Walls: {np.mean(min_wall_distances)}")  # noqa: T201
        print(  # noqa: T201
            f"Average Distance to Walls: {np.mean(average_wall_distances)}",
        )
        print(  # noqa: T201
            f"Minimum Distance to Enemies: {np.mean(min_enemy_distances)}",
        )
        print(  # noqa: T201
            f"Average Distance to Enemies: {np.mean(average_enemy_distances)}",
        )
        print(f"Average Player Health: {np.mean(healths)}")  # noqa: T201
        plot_metric(rewards, ("Episode Rewards", "Reward"))
        plot_metric(min_wall_distances, ("Minimum Distances to Walls", "Distance"))
        plot_metric(average_wall_distances, ("Average Distances to Walls", "Distance"))
        plot_metric(min_enemy_distances, ("Minimum Distances to Enemies", "Distance"))
        plot_metric(
            average_enemy_distances,
            ("Average Distances to Enemies", "Distance"),
        )
        plot_metric(healths, ("Player Health", "Health"))
        plt.show()


if __name__ == "__main__":
    # Build the argument parser and start parsing arguments
    parser = ArgumentParser(
        description="Manages the reinforcement learning training for the Hades AI"
        " agent",
    )
    parser.add_argument(
        "-w",
        "--window",
        action="store_true",
        help="Whether to show the window for the game environment or not",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        help="The level to play in the game environment",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="The seed for the game environment",
    )
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Checks if the Hades environment is valid",
    )
    build_group.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Runs the Hades AI agent using the DQN algorithm",
    )
    build_group.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Trains the Hades AI agent using the DQN algorithm",
    )
    build_group.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="Evaluates the Hades AI model to determine its performance",
    )
    args = parser.parse_args(namespace=BuildNamespace())

    # Set the environment's attributes
    ENV.show_window = args.window
    if not ENV.show_window:
        plt.switch_backend("Agg")
    if args.level:
        ENV.level = args.level
    if args.seed:
        ENV.seed = args.seed

    # Determine which argument was selected
    if args.check:
        print("*****Checking Environment*****")  # noqa: T201
        check_env(ENV)
        print("*****Checking Complete*****")  # noqa: T201
    elif args.run:
        run_dqn()
    elif args.train:
        train_dqn()
    elif args.evaluate:
        run_dqn(evaluate=True)
    ENV.close()
