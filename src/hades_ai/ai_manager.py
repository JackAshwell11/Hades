"""Manages the reinforcement learning training for the Hades AI agent."""

from __future__ import annotations

# Builtin
import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

# Pip
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.nn.functional import smooth_l1_loss
from torch.optim import AdamW

# Custom
from hades_ai.hades_environment import HadesEnvironment

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType

__all__ = ()


# The size of the input and output layers for the neural network (larger values can
# provide better results but require more computational power and may lead to
# overfitting)
FEATURE_COUNT: Final[int] = 512

# The number of episodes to train the agent for (larger values can provide better
# results but require more time)
EPISODE_COUNT: Final[int] = 2000

# The number of transitions to sample from the replay memory for each training step
# (larger values can provide more stable training but require more memory and
# computational power)
BATCH_SIZE: Final[int] = 128

# The discount factor for the Q-learning algorithm which determines the importance of
# future rewards (larger values prioritise future rewards over immediate rewards)
GAMMA: Final[float] = 0.99

# The maximum number of transitions to store in the replay memory (larger values can
# improve training but require more memory)
REPLAY_MEMORY_SIZE: Final[int] = 100000

# The number of episodes to average the rewards over for the graph (larger values can
# show a smoother graph but may hide important details)
GRAPH_MOVING_AVERAGE: Final[int] = 50

# The maximum number of steps to take in each episode before stopping (larger values can
# give the agent more time to learn but may take longer to train)
MAX_STEP_COUNT: Final[int] = 2000

# The starting epsilon value for the epsilon-greedy policy which determines the initial
# exploration rate (higher values mean the agent will explore more at the beginning of
# training which can help discover better strategies)
EPS_START: Final[float] = 0.99

# The final epsilon value for the epsilon-greedy policy which determines the minimum
# exploration rate (lower values mean the agent will exploit its learned policy more as
# training progresses focusing on the best-known actions)
EPS_END: Final[float] = 0.05

# The number of steps to decay epsilon from EPS_START to EPS_END (larger values mean the
# agent will explore more for longer before exploiting its learned policy)
EPS_DECAY: Final[int] = 1000

# The rate at which the target network's weights are adjusted towards the policy
# network's weights (smaller values mean the target network will change more slowly
# providing more stable training)
TAU: Final[float] = 0.01

# The learning rate for the AdamW optimiser which determines the step size during
# gradient descent (smaller values can lead to more stable training but may require more
# training steps)
LR: Final[float] = 0.0001

# Get the path to the output directory
OUTPUT_DIR: Final[Path] = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Store the total rewards and losses for each episode
episode_rewards: list[float] = []
episode_losses: list[float] = []


class Transition(NamedTuple):
    """Represents a transition in the replay memory.

    Attributes:
        state: The state tensor.
        action: The action tensor.
        next_state: The next state tensor.
        reward: The reward tensor.
    """

    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor


class DQN(Sequential):  # type: ignore[misc]
    """Represents a sequential deep Q-network for reinforcement learning."""

    __slots__ = ()

    def __init__(
        self: DQN,
        observation_space: torch.Space[ObsType],
        action_space: torch.Space[ActType],
    ) -> None:
        """Initialise the object.

        Args:
            observation_space: The observation space.
            action_space: The action space.
        """
        # Determine the input size based on the size of the observation space
        input_size = sum(
            np.prod(value.shape) if value.shape else 1
            for value in observation_space.values()
        )

        # Initialise the neural network creating three linear layers with ReLU
        # activation between each layer.
        # When doing this, we must be careful of overfitting for both the layer and
        # feature count as this can lead to poor performance when introducing new
        # environments.
        super().__init__(
            Linear(input_size, FEATURE_COUNT),
            ReLU(),
            Linear(FEATURE_COUNT, FEATURE_COUNT),
            ReLU(),
            Linear(FEATURE_COUNT, FEATURE_COUNT),
            ReLU(),
            Linear(FEATURE_COUNT, FEATURE_COUNT),
            ReLU(),
            Linear(FEATURE_COUNT, action_space.n),
        )


class DQNAgent:
    """Represents a deep Q-network agent for reinforcement learning."""

    __slots__ = (
        "action_space",
        "device",
        "epsilon",
        "memory",
        "optimiser",
        "policy_net",
        "steps_done",
        "target_net",
    )

    def __init__(
        self: DQNAgent,
        observation_space: torch.Space[ObsType],
        action_space: torch.Space[ActType],
    ) -> None:
        """Initialise the object.

        Args:
            observation_space: The observation space.
            action_space: The action space.
        """
        self.action_space: torch.Space[ActType] = action_space
        self.device: str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.memory: deque[Transition] = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.steps_done: int = 0
        self.policy_net: DQN = DQN(observation_space, action_space).to(self.device)
        self.target_net: DQN = DQN(observation_space, action_space).to(self.device)
        self.optimiser: AdamW = AdamW(self.policy_net.parameters(), lr=LR)
        self.epsilon: float = EPS_START

    def select_action(self: DQNAgent, state: torch.Tensor) -> torch.Tensor:
        """Select an action.

        Args:
            state: The state tensor.

        Returns:
            The action tensor.
        """
        # Increment the steps_done counter so that we can decay epsilon over time
        self.steps_done += 1

        # Select an action based on the epsilon-greedy policy
        if random.random() > self.epsilon:
            # Exploit the policy network with noise to select the best action
            with torch.no_grad():
                action = self.policy_net(state).argmax(dim=1).view(1, 1).to(self.device)
        else:
            # Explore by selecting a random action
            action = torch.tensor(
                [[self.action_space.sample()]],
                device=self.device,
                dtype=torch.float32,
            )

        # Decay epsilon after each action selection
        self.epsilon = max(EPS_END, self.epsilon - (EPS_START - EPS_END) / EPS_DECAY)
        return action

    def optimise_model(self: DQNAgent) -> float:
        """Optimise the model.

        Returns:
            The loss value.
        """
        # Get a batch of transitions
        batch = Transition(*zip(*random.sample(self.memory, BATCH_SIZE), strict=False))

        # Compute mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None],
        )
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of
        # actions taken
        state_action_values = self.policy_net(state_batch).gather(
            1,
            action_batch.long(),
        )

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
        )

        # Optimise the model
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimiser.step()
        return float(loss.item())

    def update_target_network(self: DQNAgent) -> None:
        """Update the target network."""
        # Get the state dicts for the target and policy networks
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # Perform a soft update of the target network's weights using the policy
        # network's weights
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)


def plot_graphs(*, show_result: bool = False) -> None:
    """Plot the graphs for the AI training.

    Args:
        show_result: Whether to show the final result or not.
    """

    def plot_metric(plot_num: int, metric: list[float], label: str) -> None:
        # Give the figure a number and title so that we can update it
        plt.figure(plot_num)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")

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

        # Save the graph if we're showing the final result
        if show_result:
            plt.savefig(OUTPUT_DIR / f"{label.lower()}.svg")

    # Create a graph for the losses and rewards
    plot_metric(0, episode_losses, "Loss")
    plot_metric(1, episode_rewards, "Reward")

    # Pause a bit so that plots are updated
    plt.pause(0.01)


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
        device=agent.device,
    ).unsqueeze(0)


def train_dqn(dqn_agent: DQNAgent, dqn_env: HadesEnvironment) -> None:
    """Train the DQN agent.

    Args:
        dqn_agent: The DQN agent.
        dqn_env: The DQN environment.
    """
    # Loop over the episodes
    for episode in range(EPISODE_COUNT):
        # Reset the environment
        state, _ = dqn_env.reset()

        # Ensure the state tensor includes all parts of the observation
        state = concat_observation(state)

        # Loop over the steps
        total_reward = 0.0
        total_loss = 0.0
        total_step = 0
        for _ in range(MAX_STEP_COUNT):
            # Select an action and perform it then store the reward
            action = dqn_agent.select_action(state)
            observations, reward, done, truncated, _ = dqn_env.step(action.item())
            total_reward += reward

            # Ensure the reward tensor is on the correct device
            reward = torch.tensor([reward], device=dqn_agent.device)

            # Ensure the next state tensor includes all parts of the observation
            finish = truncated or done
            next_state = concat_observation(observations) if not finish else None

            # Store the transition in memory
            dqn_agent.memory.append(Transition(state, action, next_state, reward))

            # Move to the next state
            state = next_state

            # Optimise the model if we have enough transitions
            if len(dqn_agent.memory) >= BATCH_SIZE:
                total_loss += dqn_agent.optimise_model()

            # If we're done, stop the current episode
            total_step += 1
            if finish:
                break

        # Print the episode results
        print(  # noqa: T201
            f"Finished episode {episode + 1} after {total_step} steps. Average reward:"
            f" {total_reward / total_step}, average loss: {total_loss / total_step}",
        )

        # Log the rewards and losses for this episode and plot the graphs
        episode_rewards.append(total_reward / total_step)
        episode_losses.append(total_loss / total_step)
        plot_graphs()

        # Update the target network after the episode
        dqn_agent.update_target_network()


if __name__ == "__main__":
    env = HadesEnvironment()
    agent = DQNAgent(env.observation_space, env.action_space)
    train_dqn(agent, env)
    torch.save(agent.policy_net.state_dict(), "dqn.pth")
    plot_graphs(show_result=True)
    env.close()
    input("Training complete. Press any key to exit...")


# TODO: Use argparse to add checking and running functionality
