import collections
from typing import Optional, Sequence, Any, Union, Callable

import torch as th
from torch import nn
import numpy as np

device = th.device('cpu')
def get_device():
    return device
def set_device(new_device):
    global device
    device = new_device

class Stack(nn.Module):
    def __init__(self, layers, *args, **kwargs) -> None:
        super().__init__()
        self._layers = []
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)

    def forward(self, inputs, **kwargs):
        output = inputs
        for layer in self._layers:
            output = layer(output, **kwargs)
        return output

class DenseStack(Stack):
    """
    A stack of fully connected layers. Can do batch norm and specify an alternate output activation.
    """
    def __init__(self,
                 layers: Sequence[Union[tuple, int]],
                 **kwargs) -> None:
        super(DenseStack, self).__init__()
        if layers is None:
            layers = []
        self.add(nn.Linear(*layers[0:2], **kwargs))
        self.add(nn.ReLU())
        for i in range(1,len(layers)-1):
            layer = layers[i:i+2]
            self.add(nn.Linear(*layer, **kwargs))
            self.add(nn.ReLU())

        out_layer = layers[-2:]
        self.add(nn.Linear(*out_layer, **kwargs))

class WeightNormDense(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features,out_features,bias=bias)
        self.scale = th.ones(1, out_features, requires_grad=True, device=device)

    def forward(self, inputs):
        outputs = inputs.matmul(self.weight.t())
        scale = self.scale / (th.norm(self.weight, dim=0) + 1e-8)
        outputs = outputs * scale
        if self.bias is not None:
            outputs += self.bias
        
        return outputs

class EmbeddingTranspose(nn.Module):
    """Multiply by the transpose of an embedding layer
    """
    def __init__(self, embedding_layer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = embedding_layer

    def forward(self, inputs):
        embed_mat = self.embedding.weight.detach()
        return th.matmul(inputs, embed_mat.T)

class ApplyAttentionMask(nn.Module):
    """
    Applies a mask to the attention similarities.
    """
    def __init__(self):
        super().__init__()

    def forward(self, similarity, mask=None):
        """
            Args:
                  similarity: a Tensor with shape [batch_size, heads (optional), q/k_length, q/k_length]
                  mask: a Tensor with shape [batch_size, q/k_length, q/k_length]

            Returns:
                masked_similarity: a Tensor with shape [batch_size, heads (optional), q/k_length, q/k_length]
        """
        if mask is None:
            return similarity

        # There are so many different reasons a mask might be constructed a particular manner.
        # Because of this we don't want to infer a particular construction.
        assert len(similarity.shape) in (3, 4)
        assert len(mask.shape) == 3

        # If shapes don't match, then similarity has been split for multi-headed attention
        if len(mask.shape) != len(similarity.shape):
            assert similarity[:, 0].shape == mask.shape
            mask = mask.unsqueeze(dim=1)
        else:
            assert similarity.shape == mask.shape

        # We know that we're passing this through a softmax later, thus just add a relatively large negative
        # value to mask the output avoids a hadamard product (though I think that technically it's not
        # any more efficient to do it this way operations wise)
        bias = -1e9 * th.logical_not(mask).float()
        masked_similarity = similarity + bias
        return masked_similarity

# Utility padding functions

def convert_padding_mask_to_attention_mask(sequence, padding_mask):
    """Given a padded input tensor of sequences and a boolean mask for each position
    in the sequence, returns a 3D boolean mask for use in attention.

    Args:
        sequence (th.Tensor): Tensor of shape [batch_size, sequence_length_1, ndim]
        padding_mask (th.Tensor[bool]): Tensor of shape [batch_size, sequence_length_2]
    Returns:
        th.Tensor[bool]: Tensor of shape [batch_size, sequence_length_1, sequence_length_2]
    """
    assert padding_mask.shape[0] == sequence.shape[0] and \
                                            'batch size mismatch between input sequence and  padding_mask'
    assert len(padding_mask.shape) == 2 and \
                                            'Can only convert 2D position mask to 3D attention mask'

    attention_mask = padding_mask[:, None, :].repeat(*(1, sequence.shape[1], 1))
        # 시퀀스 마스크(padding_mask)를 어텐션 마스크(attention_mask)로 변환
        # padding_mask[:, None, :]: padding_mask를 3D텐서로 확장. [batch_size, 1, sequence_length_2] 형태
        # .repeat(*(1, sequence.shape[1], 1)): [batch_size, sequence_length_1, sequence_length_2]
    
    return attention_mask   # [batch_size, sequence_length_1, sequence_length_2]

def convert_sequence_length_to_sequence_mask(sequence, sequence_lengths):
    """Given a padded input tensor of sequences and a tensor of lengths, returns
    a boolean mask for each position in the sequence indicating whether or not
    that position is padding.

    Args:
        sequence (th.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
        sequence_lengths (th.Tensor[int]): Tensor of shape [batch_size]
    Returns:
        th.Tensor[bool]: Tensor of shape [batch_size, sequence_length]
    """
    assert sequence_lengths.shape[0] == sequence.shape[0] and \
                                        'batch size mismatch between input sequence and sequence_lengths'
    assert len(sequence_lengths.shape) == 1 and \
                                        'Can only convert 1D sequence_lengths to 2D mask'

    indices = th.range(sequence.shape[1])[None, :].repeat(*(sequence_lengths.shape[0], 1))
        # th.range(sequence.shape[1]): 입력 시퀀스의 길이만큼 생성. [0, 1, 2, ..., sequence.shape[1]-1]. 시퀀스의 각 위치에 대한 인덱스
        # [None, :]: 생성된 인덱스 범위를 행렬로 변환. 1차원 벡터가 2차원 행렬로 확장. [[0, 1, 2, ..., sequence.shape[1]-1]]
        # .repeat(*(sequence_lengths.shape[0], 1)): (batch_size, sequence_length) 형태. 만약 batch_size가 3이면,
        #                                           [[0, 1, 2, ..., sequence.shape[1]-1],
        #                                            [0, 1, 2, ..., sequence.shape[1]-1],
        #                                            [0, 1, 2, ..., sequence.shape[1]-1]]
    mask = indices < sequence_lengths[:, None]
        # 각 위치의 인덱스 값이 해당 위치의 시퀀스 길이보다 작은지를 나타내는 부울(boolean) 마스크를 생성
        # True는 해당 위치가 시퀀스 길이보다 짧다는 것을 나타내고, False는 해당 위치가 시퀀스 길이보다 길거나 같다는 것
        # 시퀀스의 각 위치에 대한 패딩 여부를 나타내며, 이것이 바로 시퀀스 마스크
    return mask # [batch_size, sequence_length]

def convert_to_attention_mask(sequence, mask):
    """Automatically convert from None/1D/2D/3D mask to a boolean 3D attention mask.
    Note this does NOT allow for varying the input mask during training. We could replace
    the python if statements with tensorflow conditionals to allow this, but for the
    moment this is really a helper function and assumes that the type of mask
    passed in is fixed.

    Args:
        sequence (th.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
        mask: Optional[Tensor] of shape [batch_size]
                                     or [batch_size, sequence_length]
                                     or [batch_size, sequence_length, sequence_length]
    Returns:
        Optional[th.Tensor[bool]]: Tensor of shape [batch_size, sequence_length, sequence_length]
    """
    if mask is None:
        return None
    if len(mask.shape) == 1:
        mask = convert_sequence_length_to_sequence_mask(
            sequence, mask)
        # mask 차원이 1인 경우, 시퀀스 길이에 대한 마스크로 가정하고 시퀀스 마스크로 변환
    if len(mask.shape) == 2:
        mask = convert_padding_mask_to_attention_mask(
            sequence, mask)
        # mask 차원이 2인 경우, 패딩 마스크로 가정하고 어텐션 마스크로 변환
    if mask.dtype != th.bool:
        mask = mask.bool()

    return mask

__all__ = ['PositionEmbedding', 'EmbeddingTranspose']