import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from deeprl.infrastructure import pytorch_util as ptu
from deeprl.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 entropy_weight=0.,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight # only used for actor critic
        self.training = training
        self.nn_baseline = nn_baseline

        # Action space가 discrete 혹은 continuous에 따라 네트워크 구조 초기화.
        
        # Action이 discrete한 경우
        if self.discrete:   
            # 각 action에 대한 확률을 나타내는 로짓(logits) (정규화되지 않은 확률값)
            # logit은 확률분포의 매개변수로 사용되며, 액션을 선택할 때 확률적으로 샘플링
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        # Action space가 continuous 경우
        else:
            # action 값 자체를 출력해야 하기 때문에 action의 평균과 표준편차로 정의.
            self.logits_na = None
            # Action의 평균을 생성하는 MLP
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                          output_size=self.ac_dim,
                                          n_layers=self.n_layers,
                                          size=self.size)
            # log std 초기화.
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                                        # itertools.chain 함수를 이용해 여러 개의 파라미터 집합을 하나의 순회 가능 객체로 결합하여
                                        # optimizer에 전달해 하나의 optimizer로 여러 파라미터를 관리할 수 있다.
                                        itertools.chain([self.logstd], self.mean_net.parameters()),
                                        self.learning_rate)

        # value_function을 근사하기 위한 별도의 MLP network인 nn_baseline 초기화
        if nn_baseline:
            self.baseline = ptu.build_mlp(input_size=self.ob_dim,
                                          output_size=1,
                                          n_layers=self.n_layers,
                                          size=self.size)
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                                                self.baseline.parameters(),
                                                self.learning_rate,)
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:

        # obs의 차원을 확인하여 batch형태인지, 단일 데이터 포인트인지 결정.
        if len(obs.shape) > 1:
            observation = obs
        else:   # 단일 데이터 포인트일 경우, 차원을 추가하여 batch형태로 만듦.
            observation = obs[None]

        # Numpy 배열을 PyTorch 텐서로 변환.
        observation = ptu.from_numpy(observation)
        # action = self(observation)

        # observation을 self()을 통해 모델에 전달하여 행동 분포를 계산.
        # self()는 'forward' 메소드를 내부적으로 호출
        action_distribution = self(observation)
        
        # sample() 함수를 이용해 확률분포에서 무작위로 추출.
        action = action_distribution.sample()  # don't bother with rsample
        
        # Pytorch 텐서에서 다시 Numpy 배열로 변환.
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function outputs a distribution object representing the policy output for the particular observations.
    # assumes first dimension of observations 
    def forward(self, observation: torch.FloatTensor):
        
        # Action space가 discrete 경우
        if self.discrete:
            # observation관측치를 network 통과시켜 logit 계산.
            logits = self.logits_na(observation)
            # logits을 사용해서 카테고리컬 분포를 생성하고 반환.
            # torch.distributions.Categorical은 다항분포로 이산적인 action space에서
            # 각 action에 대한 확률을 정의. 주어진 확률(logits)에 기반하여 action을 무작위 샘플링
            return torch.distributions.Categorical(logits=logits)
        
        # Action space가 continuous 경우
        else:
            # observation관측치를 통해 action의 평균값 계산.
            batch_mean = self.mean_net(observation)
            # 로그 표준편차 특정 범위 내로 제한.
            logstd = torch.clamp(self.logstd, -10, 2) 
            # 표준편차를 사용해 공분산 행렬을 생성.
            # action간의 관계와 변동성을 고려하기 위해 공분산 행렬을 사용.
            scale_tril = torch.diag(torch.exp(logstd))
            # batch 크기를 계산.
            batch_dim = batch_mean.shape[0]
            # 각 batch item에 대해 공분산 행렬을 반복.
            # 각 배치에 대해 공분산 행렬을 반복하는 이유는 배치 처리 중 각 데이터 포인트(observations)가 
            # 독립적으로 처리되어 개별적인 액션 분포를 계산해야 하기 때문. 
            # 이를 통해 각 관측치에 대한 액션의 분포를 별도로 모델링할 수 있습니다.
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            # 평균과 공분산 행렬을 사용해서 다변량 정규분포를 생성하고 반환.
            return distributions.MultivariateNormal(
                                                    batch_mean,
                                                    scale_tril=batch_scale_tril,)

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None):
        
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        """
        TODO: compute the behavior cloning loss by maximizing the log likelihood
        expert actions under the policy.
        Hint: look at the documentation for torch.distributions 
        """
        action_distribution = self(observations)
        log_prob = action_distribution.log_prob(actions)
        loss = -log_prob.mean()
        """
        END CODE
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, acs_na, adv_n=None, acs_labels_na=None,
                   qvals=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(acs_na)
        adv_n = ptu.from_numpy(adv_n)
        """
        TODO: compute the policy gradient given already compute advantages adv_n
        """
        loss = - (self(observations).log_prob(actions) * adv_n).mean()
        """
        END CODE
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            # first standardize qvals to make this easier to train
            targets_n = (qvals - np.mean(qvals)) / (np.std(qvals) + 1e-8)
            targets_n = ptu.from_numpy(targets_n)
            """
            TODO: update the baseline value function by regressing to the values
            Hint: see self.baseline_loss for the appropriate loss
            """
            baseline_loss = None
            """
            END CODE
            """
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        else:
            baseline_loss = None
        return {
            'Training Loss': ptu.to_numpy(loss),
            'Baseline Loss': ptu.to_numpy(baseline_loss) if baseline_loss else 0,
        }

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array
            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]
        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())

class MLPPolicyAC(MLPPolicy):
    def forward(self, observations: torch.FloatTensor):
        if self.discrete:
            return super(MLPPolicyAC).forward(observations)
        else:
            base_dist = super(MLPPolicyAC, self).forward(observations)
            # for AC methods, we need to ensure actions sampled from the env
            # are valid actions for the environment. 
            # Since the action spaces are bounded between [-1, 1], we apply
            # a tanh function to the Gaussian policy samples.
            return torch.distributions.transformed_distribution.TransformedDistribution(
                    base_dist, [torch.distributions.transforms.TanhTransform()])


    def update(self, observations, critic):
        observations = ptu.from_numpy(observations)
        """
        TODO: implement policy loss with learned critic. Update the policy 
        to maximize the expected Q value of actions, using a single sample
        from the policy for each state.
        Hint: assuming we are in continous action spaces and using Gaussian 
        distributions, look at the rsample function to differentiate through 
        samples from the action distribution.
        """
        loss = None
        """
        END CODE
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'Actor Training Loss': ptu.to_numpy(loss),
        }