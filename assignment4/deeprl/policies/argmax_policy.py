import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        """
        TODO: return the action that maximizes the Q-value 
        at the current observation as the output.
        """
        # argmax() : 주어진 축에 대해 최대값을 가지는 인덱스 반환
        # -1 : 텐서의 마지막 차원을 의미. 
        action = self.critic.qa_values(observation).argmax(-1)
        """
        END CODE
        """
        return action.squeeze()
