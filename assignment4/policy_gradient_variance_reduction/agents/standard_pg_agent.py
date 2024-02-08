import torch
from .base_agent import BaseAgent

class StandardPGAgent(BaseAgent):
    def learn(self, states, actions, rewards):
        policy_gradient = []
        for log_prob, reward in zip(actions, rewards):
            reward = torch.tensor(reward, dtype=torch.float32)
            policy_gradient.append(-log_prob * reward.unsqueeze(0))  # 차원 추가

        self.optimizer.zero_grad()
        policy_gradient = torch.cat(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()
