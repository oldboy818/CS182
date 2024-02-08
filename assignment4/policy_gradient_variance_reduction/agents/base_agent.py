import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BaseAgent:
    def __init__(self, input_size, output_size, learning_rate=1e-2):
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=0)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.model(state)
        action = np.random.choice(len(action_probs), p=np.squeeze(action_probs.detach().numpy()))
        return action

    def learn(self, states, actions, rewards):
        raise NotImplementedError
