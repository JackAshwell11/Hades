from pathlib import Path

import torch
import torch.nn.functional as F
from hades_environment import HadesEnvironment
from torch import nn

# Load the environment
env = HadesEnvironment()

# Get the number of actions and observations
n_actions = env.action_space.n


# Define the DQN class (same as in your training script)
class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        # Correct input size calculation based on observation space
        self.input_size = 10

        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)


state, info = env.reset()
n_observations = len(state)

# Initialize the model and load the saved weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load(Path(__file__).parent.joinpath("dqn_model.pth")))
model.eval()

# Prepare the input data (example state)
import numpy as np

state = torch.tensor(
    np.concatenate(
        (
            state["current_position"],
            state["current_velocity"],
            state["distance_to_walls"],
            # state["raycasts"],
            state["goal_position"],
        ),
    ),
    dtype=torch.float32,
    device=device,
).unsqueeze(0)

# Perform inference
while True:
    with torch.no_grad():
        action = model(state).max(1)[1].item()

    print(f"Selected action: {action}")

    # Apply the action in the environment
    state, reward, done, _, _ = env.step(action)
    state = torch.tensor(
        np.concatenate(
            (
                state["current_position"],
                state["current_velocity"],
                state["distance_to_walls"],
                # state["raycasts"],
                state["goal_position"],
            ),
        ),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    env.render()
