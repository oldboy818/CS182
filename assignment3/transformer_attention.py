from typing import Optional, Callable, Tuple

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
# from tensorflow.keras.layers import Layer

from transformer_utils import ApplyAttentionMask

class AttentionQKV(nn.Module):
    """
    Computes attention based on provided similarity metric.
    """

    def __init__(self):
        super().__init__()
        self.apply_mask = ApplyAttentionMask()

    def forward(self, queries, keys, values, mask=None):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_queries]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        ####################################  YOUR CODE HERE  ####################################
        # n_queries corresponds to the sequence length on the query side
        # n_keyval corresponds to the sequence length on the key side (and value, as they are one and the same)
        # depth_k is the size of the projection that the key / query comparison is performed on.
        # depth_v is the size of the projection of the value projection. In a setting with one head, 
        # it is usually the dimension (dim) of the Transformer.
        # heads corresponds to the number of heads the attention is performed on.
        # If you are unfamiliar with attention heads, read section 3.2.2 of the Attention is all you need paper
         
        # PART 1: Implement Attention QKV
        # Use queries, keys and values to compute the output of the QKV attention

        # As defined is the Attention is all you need paper: https://arxiv.org/pdf/1706.03762.pdf
        key_dim = th.tensor(keys.shape[-1],dtype=th.float32)
        similarity = th.matmul(queries, keys.transpose(-2, -1)) / th.sqrt(key_dim)
                    # Compute the similarity according to the QKV formula

        masked_similarity = self.apply_mask(similarity, mask=mask) 
                            # We give you the mask to apply so that it is correct, you do not need to modify this.
        weights = th.nn.functional.softmax(masked_similarity, dim=-1)
                # Turn the similarity into a normalized output. Remember that the last dim contains the features
        output = th.matmul(weights, values)
                # Obtain the output
        ####################################  END OF YOUR CODE  ##################################

        return output, weights

class MultiHeadProjection(nn.Module):

    def __init__(self, n_heads, feature_sizes):
        """Map the multi-headed attention across the map

        Arguments:
            n_heads {int} -- The number of heads in the attention map
            feature_sizes {int} -- The size of the feature dimensions for key, query, and value
        """

        super().__init__()
        self.attention_map = AttentionQKV()
        self.n_heads = n_heads

        for size in feature_sizes:
            assert size % self.n_heads == 0, 'Shape of feature input must be divisible by n_heads'

    def forward(self, inputs, mask=None):
        """Fast multi-head attention.

        :param queries: Tensor with shape [batch_size, n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, n_keyval, depth_v]

        :return: output: Tensor with shape [batch_size, n_queries, depth_v]
        """
        queries, keys, values = inputs

        # Split each of the projection into its heads, by adding a new dimension
        # You must implement _split_heads, and _combine_heads
        queries_split = self._split_heads(queries)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        # Apply the attention map
        attention_output_split, _ = self.attention_map(queries_split, keys_split, values_split, mask=mask)

        # Re-combine the heads together, and return the output.
        output = self._combine_heads(attention_output_split)
        return output

    def _split_heads(self, tensor):
        assert len(tensor.shape) == 3
        ####################################  YOUR CODE HERE  ####################################
        # PART 2: Implement the Multi-head attention.
        # You are given a Tensor which is one of the projections (K, Q or V)
        # and you must "split it" in self.n_heads. This splitting should add a dimension to the tensor,
        # so that each head acts independently

        batch_size, tensorlen = tensor.shape[0], tensor.shape[1]
        feature_size = tensor.shape[2]
                # tensorlen은 'sequence_lenghth'를 의미
                # feature_size는 임베딩 벡터의 차원을 의미

        new_feature_size = feature_size // self.n_heads
                        # Compute what the feature size per head is.
                        # featute_size = n_heads * new_feature_size이기 때문
        
        tensor = tensor.reshape(batch_size, tensorlen, self.n_heads, new_feature_size)
                # Reshape this projection tensor so that it has n_heads, each of new_feature_size
        
        tensor = tensor.permute(0, 2, 1, 3)
            # Transpose the matrix so the outer-dimensions are the batch-size and the number of heads
            # shape (batch_size, n_heads, tensorlen, new_feature_size)
            # 이로써 각 헤드가 독립적으로 시퀀스에 대해서 연산을 수행할 수 있다
        return tensor
        ##########################################################################################

    def _combine_heads(self, tensor):
        assert len(tensor.shape) == 4
        ####################################  YOUR CODE HERE  ####################################
        # PART 2: Implement the Multi-head attention.
        # You are given the output from all the heads, and you must combine them back into 1 rank-3 matrix

        # Transpose back compared to the split, so that the outer dimensions are batch_size and sequence_length again
        tensor = tensor.permute(0, 2, 1, 3)
        batch_size, tensorlen = tensor.shape[0], tensor.shape[1]
        feature_size = tensor.shape[-1]

        new_feature_size = self.n_heads * feature_size
                        # What is the new feature size, if we combine all the heads
        tensor = tensor.reshape(batch_size, tensorlen, new_feature_size)
                # Reshape the Tensor to remove the heads dimension and come back to a Rank-3 tensor
        return tensor
        ##########################################################################################

class MultiHeadAttention(nn.Module):
    """
    Fast multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, n_heads, input_shapes):
        super().__init__()
            # 상위 클래스(nn.Module의 생성자)를 초기화.
            # 이는 PyTorch 모듈의 기본 설정을 활성화하여 클래스가 정상적으로 동작할 수 있게 합니다

        self.qa_channels, self.ma_channels = input_shapes
            # 쿼리와 메모리 입력의 채널 수

        self.n_heads = n_heads
        self.attention_layer = MultiHeadProjection(n_heads, (self.qa_channels,self.ma_channels))
                            # (멀티헤드 프로젝션) 어텐션 레이어 초기화

        # 입력크기 검증. 
        assert self.qa_channels % self.n_heads == 0 and self.ma_channels % self.n_heads == 0 and \
                                                        'Feature size must be divisible by n_heads'
        assert self.qa_channels == self.ma_channels and 'Cannot combine tensors with different shapes'

        # 선형 레이어와 가중치 초기화. 가중치 정규화는 (가중치 벡터의 크기를 조절하여) 학습 중에 가중치가 특정 범위를 벗어나지 않게 함
        self.query_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
        self.key_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
        self.value_layer = weight_norm(nn.Linear(self.ma_channels, self.ma_channels, bias=False))
        
        self.output_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
                            # nn.Linear : 입력과 출력 차원을 동일하게

        # Xavier uniform. 신경망의 가중치를 적절한 분산으로 초기화하여 학습 안정화
        def weights_init(m):
            # if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
        
        self.query_layer.apply(weights_init)
        self.key_layer.apply(weights_init)
        self.value_layer.apply(weights_init)
        self.output_layer.apply(weights_init)


    def forward(self, inputs, mask=None):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        assert (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 2 and \
                                                        'Must pass query and memory'
        query_antecedent, memory_antecedent = inputs
            # query_antecedent: query 벡터를 생성하기 위한 입력 텐서
            # memory_antecedent: key와 value 벡터를 생성하기 위한 입력 텐서

            # 두 antecedent로 구분하는 이유는 Transformer의 두 종류 어텐션 메커니즘에서 다른 역할을 수행하기 위함
            # self-attention과 cross-attention.
            # 쿼리 앤테세던트는 어텐션을 요청하는 위치, 메모리 앤테세던트는 어텐션을 제공하는 위치
        q = self.query_layer(query_antecedent)
        k = self.key_layer(memory_antecedent)
        v = self.value_layer(memory_antecedent)

        attention_output = self.attention_layer((q, k, v), mask = mask)
        output = self.output_layer(attention_output)
        return output