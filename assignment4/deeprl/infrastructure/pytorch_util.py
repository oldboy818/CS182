from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    # 매개변수 타입 체크. 활성화 함수가 문자열로 전달되면, 해당하는 Pytorch 모듈로 변환
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    
    # 신경망 층 리스트 생성. 신경망의 각 층을 순서대로 포함하게 된다.
    layers = []
    in_size = input_size
    
    # Hidden layer 추가 루프.
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    # 출력층 및 출력층 활성화 함수 추가
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    
    # nn.Sequential을 이용해서 리스트에 추가된 모든 층을 연결한 신경망을 반환.
    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    device = torch.device("cpu")
    print("Using CPU for this assignment. There may be some bugs with using GPU that cause test cases to not match. You can uncomment the code below if you want to try using it.")
    # if torch.cuda.is_available() and use_gpu:
        # device = torch.device("cuda:" + str(gpu_id))
        # print("Using GPU id {}".format(gpu_id))
    # else:
        # device = torch.device("cpu")
        # print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
