"""Manages the deep Q-network agent for reinforcement learning."""

from __future__ import annotations

# Builtin
import random
from collections import deque
from typing import TYPE_CHECKING, Final, NamedTuple

# Pip
import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.nn.functional import smooth_l1_loss
from torch.optim import AdamW

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType

__all__ = (
    "BATCH_SIZE",
    "DQN",
    "EPS_DECAY",
    "EPS_END",
    "EPS_START",
    "FEATURE_COUNT",
    "GAMMA",
    "LR",
    "REPLAY_MEMORY_SIZE",
    "TAU",
    "DQNAgent",
    "Transition",
)

# The size of the input and output layers for the neural network (larger values can
# provide better results but require more computational power and may lead to
# overfitting)
FEATURE_COUNT: Final[int] = 512

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

        # Initialise the neural network creating two linear layers with ReLU activation
        # between each layer.
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
    """Represents a deep Q-network agent for reinforcement learning.

    Attributes:
        device: The device to use for computations (CPU or GPU).
        memory: The replay memory to store transitions.
        policy_net: The policy network for action selection.
        target_net: The target network for stable Q-learning.
        optimiser: The optimiser for training the policy network.
        epsilon: The exploration rate for the epsilon-greedy policy.
    """

    __slots__ = (
        "action_space",
        "device",
        "epsilon",
        "memory",
        "optimiser",
        "policy_net",
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
        self.policy_net: DQN = DQN(observation_space, action_space).to(self.device)
        self.target_net: DQN = DQN(observation_space, action_space).to(self.device)
        self.optimiser: AdamW = AdamW(self.policy_net.parameters(), lr=LR)
        self.epsilon: float = EPS_START

    def select_action(
        self: DQNAgent,
        state: torch.Tensor,
        *,
        train: bool,
    ) -> torch.Tensor:
        """Select an action.

        Args:
            state: The state tensor.
            train: Whether to train the agent or not.

        Returns:
            The action tensor.
        """
        # Select an action based on the epsilon-greedy policy
        if random.random() > self.epsilon or not train:
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

        # Decay epsilon linearly after each action selection
        self.epsilon = max(EPS_END, self.epsilon - (EPS_START - EPS_END) / EPS_DECAY)
        return action

    def optimise_model(self: DQNAgent) -> float:
        """Optimise the model.

        Returns:
            The loss value.
        """
        # Get a random batch of transitions
        batch = Transition(*zip(*random.sample(self.memory, BATCH_SIZE), strict=False))

        # Identify non-terminal states and combine the corresponding batch elements
        non_terminal_states_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )
        non_terminal_next_states = torch.cat(
            [s for s in batch.next_state if s is not None],
        )
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))

        # Compute the predicted Q-values for the current states and the actions taken
        predicted_q_values = self.policy_net(state_batch).gather(
            1,
            action_batch.long(),
        )

        # Compute the expected Q-values (using the Bellman equation) for the next states
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_terminal_states_mask] = (
                self.target_net(non_terminal_next_states).max(1).values
            )
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute the loss between the predicted and expected Q-values using the Huber
        # loss (which is less sensitive to outliers and provides more stable training)
        loss = smooth_l1_loss(
            predicted_q_values,
            expected_state_action_values.unsqueeze(1),
        )

        # Reset the gradients and perform backpropagation
        self.optimiser.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients and update the policy network
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
