import torch
from .base_agent import BaseAgent

class RTGPGAgent(BaseAgent):
    def learn(self, states, actions, rewards):
        policy_gradient = []
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + (G * 0.99)  # Discount factor of 0.99
            policy_gradient.insert(0, -actions[i] * G)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()