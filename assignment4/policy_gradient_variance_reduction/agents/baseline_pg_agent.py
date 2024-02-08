import torch
from .base_agent import BaseAgent

class BaselinePGAgent(BaseAgent):
    def learn(self, states, actions, rewards):
        policy_gradient = []
        baseline = sum(rewards) / len(rewards)
        for log_prob, reward in zip(actions, rewards):
            advantage = reward - baseline  # Subtracting the baseline from the reward
            policy_gradient.append(-log_prob * advantage)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()
