"""Manages the reinforcement learning training for the Hades AI agent."""

from __future__ import annotations

# Builtin
import csv
from argparse import ArgumentParser, Namespace
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Final

# Pip
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from matplotlib import get_backend

# Custom
from hades_ai.dqn import DQNAgent, Transition
from hades_ai.hades_environment import HadesEnvironment

if TYPE_CHECKING:
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
MAX_STEP_COUNT: Final[int] = 1000

# The number of runs to perform for the grid search (larger values can provide better
# accuracy but require more time)
GRID_SEARCH_RUNS: Final[int] = 3

# The interval at which to save the gameplay and graphs (larger values can reduce the
# number of saves, but may miss important details)
SAVE_INTERVAL: Final[int] = 50

# The name of the model file that will be saved
MODEL_NAME: Final[str] = "model.pth"

# The name of the CSV file that will be saved for the grid search
GRID_SEARCH_CSV: Final[str] = "grid_search.csv"

# Get the path to the output directory
ROOT_OUTPUT_DIR: Final[Path] = Path(__file__).parent / "output"
ROOT_OUTPUT_DIR.mkdir(exist_ok=True)

# Create the environment for the AI agent
ENV: Final[HadesEnvironment] = HadesEnvironment()

# The hyperparameters for the DQN agent
HYPERPARAMETERS: Final[dict[str, float | int]] = {
    "LAYER_COUNT": 3,
    "FEATURE_COUNT": 256,
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "REPLAY_MEMORY_SIZE": 100000,
    "EPS_START": 0.99,
    "EPS_END": 0.05,
    "EPS_DECAY": 1000,
    "TAU": 0.01,
    "LR": 0.0001,
}

# The grid of hyperparameters to search over
HYPERPARAMETERS_GRID: Final[dict[str, list[float | int]]] = {
    "LAYER_COUNT": [2, 3, 4],
    "FEATURE_COUNT": [128, 256, 512],
    "BATCH_SIZE": [64, 128, 256],
    "GAMMA": [0.95, 0.99, 0.999],
    "REPLAY_MEMORY_SIZE": [50000, 100000, 200000],
    "EPS_START": [0.99, 0.9, 0.8],
    "EPS_END": [0.01, 0.05, 0.1],
    "EPS_DECAY": [500, 1000, 2000],
    "TAU": [0.001, 0.01, 0.1],
    "LR": [0.0001, 0.001, 0.01],
}

# Check if we're running in interactive mode or not
if IS_IPYTHON := "inline" in get_backend():
    from IPython import display


class BuildNamespace(Namespace):
    """Allows typing of an argparse namespace for the CLI."""

    check: bool
    run: bool
    train: bool
    grid: bool


def plot_graphs(save_dir: Path | None, *results: tuple[str, list[float]]) -> None:
    """Plot the graphs for the AI training.

    Args:
        save_dir: The directory to save the graphs to.
        results: The results to plot.
    """

    def plot_metric(plot_num: int, metric: list[float], label: str) -> None:
        # Give the figure a number and title so that we can update it
        plt.figure(plot_num)
        plt.clf()
        plt.title("Training")

        # Label the x and y axes and plot the metric
        plt.xlabel("Episode")
        plt.ylabel(label)
        plt.plot(metric, label=label)

        # Plot the moving average of the metric if we have enough data
        if len(metric) >= GRAPH_MOVING_AVERAGE:
            metric_t = torch.tensor(metric, dtype=torch.float32)
            plt.plot(
                metric_t.unfold(0, GRAPH_MOVING_AVERAGE, 1).mean(1).numpy(),
                label=f"Average {label.lower()}",
            )

        # Add a legend to the graph after plotting the metric (we can't add it before as
        # we need to know the label)
        plt.legend()

        # Save the graph if possible
        if save_dir:
            plt.savefig(save_dir / f"{label.lower()}.png")

    # Create a graph for the losses and rewards
    for i, result in enumerate(results):
        plot_metric(i, result[1], result[0])

    # Pause a bit so that plots are updated
    plt.pause(0.01)

    # If we're running in interactive mode, update the graphs
    if IS_IPYTHON:
        display.display(plt.gcf())
        display.clear_output(wait=True)


def concat_observation(agent: DQNAgent, obs: ObsType) -> torch.Tensor:
    """Concatenate the observation into a single tensor.

    Args:
        agent: The DQN agent to concatenate the observation for.
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
        device=agent.device,
    ).unsqueeze(0)


def process_episode(agent: DQNAgent, *, train: bool = True) -> tuple[float, float, int]:
    """Process an episode of the environment.

    Args:
        agent: The DQN agent to process the episode with.
        train: Whether to train the agent or not.

    Returns:
        A tuple containing the total reward, total loss, and total step count.
    """
    # Reset the environment
    state, _ = ENV.reset()

    # Ensure the state tensor includes all parts of the observation
    state = concat_observation(agent, state)

    # Run the episode and get the total reward, loss, and step count
    total_reward = 0
    total_loss = 0
    total_step = 0
    for _ in range(MAX_STEP_COUNT):
        # Select an action based on whether we're training or not then perform it and
        # get the next state
        action = agent.select_action(state, train=train)
        observation, reward, done, truncated, _ = ENV.step(action.item())
        total_reward += reward
        finish = truncated or done
        next_state = concat_observation(agent, observation) if not finish else None

        # Store the transition in memory
        if train:
            agent.memory.append(
                Transition(
                    state,
                    action,
                    next_state,
                    torch.tensor([reward], device=agent.device),
                ),
            )

        # Move to the next state
        state = next_state

        # Optimise the model if we have enough transitions
        if train and len(agent.memory) >= agent.hyperparams["BATCH_SIZE"]:
            total_loss += agent.optimise_model()

        # If we're done, stop the current episode
        total_step += 1
        if finish:
            break
    return total_reward, total_loss, total_step


def train_dqn(agent: DQNAgent, output_dir: Path) -> None:
    """Train the DQN agent in the environment.

    Args:
        agent: The DQN agent to train.
        output_dir: The directory to save the training data to
    """
    # Loop over the episodes
    episode_rewards = []
    episode_losses = []
    for episode in range(EPISODE_COUNT):
        # Enable saving if possible
        episode_dir = ROOT_OUTPUT_DIR / f"{episode + 1}"
        episode_dir.mkdir(exist_ok=True)
        if episode % SAVE_INTERVAL == 0 or episode == EPISODE_COUNT - 1:
            ENV.window.make_writer(episode_dir / f"episode_{episode}.mp4")

        # Loop over the steps
        total_reward, total_loss, total_step = process_episode(agent)

        # Print the episode results
        print(  # noqa: T201
            f"Finished episode {episode + 1} (saving: {ENV.window.writer is not None})"
            f" after {total_step} steps, average reward: {total_reward / total_step},"
            f" average loss: {total_loss / total_step}",
        )

        # Log the rewards and losses for this episode
        episode_rewards.append(total_reward)
        episode_losses.append(total_loss)

        # Plot the graphs for the episode and save them, the video, and the model if
        # possible
        save_dir = episode_dir if ENV.window.writer else None
        plot_graphs(save_dir, ("Reward", episode_rewards), ("Loss", episode_losses))
        if save_dir:
            ENV.window.save_video()
            torch.save(agent.policy_net.state_dict(), episode_dir / MODEL_NAME)

        # Update the target network after the episode
        agent.update_target_network()

    # Save the final model
    torch.save(agent.policy_net.state_dict(), output_dir / MODEL_NAME)


def run_dqn(agent: DQNAgent) -> None:
    """Run the DQN agent in the environment.

    Args:
        agent: The DQN agent to run.
    """
    # Find the most recent model
    path = (
        max(
            (d for d in ROOT_OUTPUT_DIR.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            default=ROOT_OUTPUT_DIR,
        )
        / MODEL_NAME
    )
    if not path.exists():
        print("No model found, please train the model first")
        return

    # Load the trained model then run the agent
    agent.policy_net.load_state_dict(torch.load(path, weights_only=True))
    total_reward, _, total_step = process_episode(agent, train=False)
    print(f"Average reward: {total_reward / total_step}")


def grid_search() -> None:
    """Perform a grid search to find the best hyperparameters for the DQN."""
    # Prepare the CSV file
    with open(ROOT_OUTPUT_DIR / GRID_SEARCH_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameters", "Run", "Reward", "Steps"])

    # Loop over all the hyperparameter combinations
    combinations = list(product(*HYPERPARAMETERS_GRID.values()))
    for i, combination in enumerate(combinations):
        params = dict(zip(HYPERPARAMETERS_GRID.keys(), combination, strict=False))
        for run in range(GRID_SEARCH_RUNS):
            # Train the model with the current hyperparameters
            print(
                f"Grid Search {i + 1}/{len(combinations)} - Parameters: {params} - Run: {run + 1}",
            )
            grid_agent = DQNAgent(ENV.observation_space, ENV.action_space, params)
            param_str = "_".join(f"{value}" for value in params.values())
            output_dir = ROOT_OUTPUT_DIR / f"grid_search_{param_str}_{run}"
            output_dir.mkdir(exist_ok=True)
            train_dqn(grid_agent, output_dir)

            # Evaluate the trained model and log the results
            total_reward, _, total_steps = process_episode(grid_agent, train=False)
            with open(ROOT_OUTPUT_DIR / GRID_SEARCH_CSV, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [param_str, run + 1, np.mean(total_reward), total_steps],
                )
            progress = (
                (i * GRID_SEARCH_RUNS + run + 1)
                / (len(combinations) * GRID_SEARCH_RUNS)
                * 100
            )
            print(f"Grid Search Progress: {progress:.2f}%")


if __name__ == "__main__":
    # Build the argument parser and start parsing arguments
    parser = ArgumentParser(
        description="Manages the reinforcement learning training for the Hades AI"
        " agent",
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
        "-g",
        "--grid",
        action="store_true",
        help="Runs a grid search to find the best hyperparameters for the DQN",
    )
    args = parser.parse_args(namespace=BuildNamespace())

    # Determine which argument was selected
    agent = DQNAgent(ENV.observation_space, ENV.action_space, HYPERPARAMETERS)
    if args.check:
        print("*****Checking Environment*****")
        check_env(ENV)
        print("*****Checking Complete*****")
    elif args.run:
        run_dqn(agent)
    elif args.train:
        train_dqn(agent, ROOT_OUTPUT_DIR)
    elif args.grid:
        grid_search()
    ENV.close()
