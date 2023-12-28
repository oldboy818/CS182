from typing import Optional, List
from collections import namedtuple

import torch as th
from torch import nn
from torch.nn import functional as F

import transformer_utils
from transformer_utils import EmbeddingTranspose, get_device
from transformer_attention import MultiHeadAttention

class PositionEmbedding(nn.Module):
    """
    Adds positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self, hidden_size):
        super(PositionEmbedding, self).__init__()

        assert hidden_size % 2 == 0 and 'Model vector size must be even for sinusoidal encoding'
        power = th.arange(0, hidden_size, step=2, dtype=th.float32)[:] / hidden_size
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def forward(self, inputs, start=1):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        ####################################  YOUR CODE HERE  ####################################
        # PART 3: Implement the Position Embedding.
        # As stated in section 3.5 of the paper, attention does not naturally embed position information
        # To incorporate that, the authors use a variable frequency sin embedding.
        # Note that we use zero-indexing here while the authors use one-indexing

        assert inputs.shape[-1] == self.hidden_size and 'Input final dim must match model hidden size'

        batch_size = inputs.shape[0]
        sequence_length = inputs.shape[1]

        # obtain a sequence that starts at `start` and increments for `sequence_length `
        seq_pos = th.arange(start, sequence_length + start, dtype=th.float32)
                # 시퀀스 위치 생성 : start부터 시퀀스 길이 + start까지 연속적인 숫자 생성
                # 생성된 숫자는 float32 즉, 부동소수점 형태
        seq_pos_expanded = seq_pos[None, :, None]
                # 시퀀스 위치 확장 : 3차원 텐서, 첫 번째/세 번째 차원은 단일 요소, 두 번째 차원은 시퀀스 길이
        index = seq_pos_expanded.repeat(*[1, 1, self.hidden_size//2])
                # 텐서를 반복하여 확장. 첫 번째, 두 번째 차원은 반복 X
                # 세 번째 차원은 self.hidden_size//2 만큼 반복

        # create the position embedding as described in the paper
        # use the `divisor` attribute instantiated in __init__ 
        sin_embedding = th.sin(index / self.divisor)
        cos_embedding = th.cos(index / self.divisor)

        # interleave the sin and cos. For more info see:
        # https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/3
        position_shape = (1, sequence_length, self.hidden_size) # fill in the other two dimensions
        position_embedding = th.stack((sin_embedding, cos_embedding), dim=3).view(position_shape)
                            # sin/cos_embedding은 3차원 텐서 [batch_size, sequence_length, hidden_size // 2]
                            # dim=3 : sin/cos 두 텐서가 새로운 4번째 차원을 기준으로 stacking
                            # 그 결과, [batch_size, sequence_length, hidden_size // 2, 2]의 4차원 텐서

        pos_embed_deviced = position_embedding.to(get_device())
        return  inputs + pos_embed_deviced  # add the embedding to the input
        ####################################  END OF YOUR CODE  ##################################

class TransformerFeedForward(nn.Module):
    def __init__(self, input_size,
                 filter_size,
                 hidden_size,
                 dropout):
        super(TransformerFeedForward, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.feed_forward = nn.Sequential(
                                nn.Linear(input_size, filter_size),
                                nn.ReLU(),
                                nn.Linear(filter_size, hidden_size)
                            )
        # Xavier uniform. 신경망의 가중치를 적절한 분산으로 초기화하여 학습 안정화
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.feed_forward.apply(weights_init)
        self.dropout = nn.Dropout(0 if dropout is None else dropout)

    def forward(self, inputs):
        ####################################  YOUR CODE HERE  ####################################
        # PART 4.1: Implement the FeedForward Layer.
        # As seen in fig1, the feedforward layer includes a normalization and residual
        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        dense_drop = self.dropout(dense_out) # Add the dropout here
        return dense_drop + inputs # Add the residual here
        ####################################  END OF YOUR CODE  ##################################

class TransformerEncoderBlock(nn.Module):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

    def __init__(self,
                 input_size,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout = None) -> None:
        super(TransformerEncoderBlock, self).__init__()

        self.norm = nn.LayerNorm(input_size)
        self.self_attention = MultiHeadAttention(n_heads,[input_size, input_size])
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def forward(self, inputs, self_attention_mask=None):
        ####################################  YOUR CODE HERE  ####################################
        # PART 4.2: Implement the Transformer Encoder according to section 3.1 of the paper.
        # Perform a multi-headed self-attention across the inputs.

        # First normalize the input with the LayerNorm initialized in the __init__ function (self.norm)
        norm_inputs = self.norm(inputs)

        # Apply the self-attention with the normalized input, use the self_attention mask as the optional mask parameter.
        attn = self.self_attention(inputs = (norm_inputs, norm_inputs), mask=self_attention_mask)
                # head 수가 2이기 때문에 input 역시 2차원
        
        # Apply the residual connection. res_attn should sum the attention output and the original, non-normalized inputs
        res_attn = attn + inputs # Residual connection of the attention block

        # output passes through a feed_forward network
        output = self.feed_forward(res_attn)

        return output

class TransformerDecoderBlock(nn.Module):
    """A decoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: two Tensors encoder_outputs, decoder_inputs
                    encoder_outputs -> a Tensor with shape [batch_size, sequence_length, channels]
                    decoder_inputs -> a Tensor with shape [batch_size, decoding_sequence_length, channels]

    :return: output: Tensor with same shape as decoder_inputs
    """

    def __init__(self,
                 input_size,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout = None) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(input_size)
        self.self_attention = MultiHeadAttention(n_heads,[input_size, input_size])

        self.cross_attention = MultiHeadAttention(n_heads,[input_size, input_size])
        self.cross_norm_source = nn.LayerNorm(input_size)
        self.cross_norm_target = nn.LayerNorm(input_size)
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def forward(self, decoder_inputs, encoder_outputs, self_attention_mask=None, cross_attention_mask=None):    
        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        ####################################  YOUR CODE HERE  ####################################
        # PART 4.2: Implement the Transformer Decoder according to section 3.1 of the paper.
        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        # Compute the self-attention over the decoder inputs. This uses the self-attention
        # mask to control for the future outputs.
        # This generates a tensor of size [batch_size x target_len x d_model]

        # (Masked Multi-Head) Self-Attention
        norm_decoder_inputs = self.self_norm(decoder_inputs)
                            # Layer Normalization 적용. Batch norm과 유사하지만, not across the batch
        target_selfattn = self.self_attention(inputs = (norm_decoder_inputs, norm_decoder_inputs),
                                              mask = self_attention_mask)
                        # Head 수(2)에 맞춰 입력도 2차원
        res_target_self_attn = target_selfattn + decoder_inputs
                            # Residual connection of the self-attention block

        # Compute the attention using the keys/values from the encoder, and the query from the
        # decoder. This takes the encoder output of size [batch_size x source_len x d_model] and the
        # target self-attention layer of size [batch_size x target_len x d_model] and then computes
        # a multi-headed attention across them, giving an output of [batch_size x target_len x d_model]
        # using the encoder as the keys and values and the target as the queries

        # Cross-Attention
        norm_target_selfattn = self.cross_norm_target(res_target_self_attn)
        norm_encoder_outputs = self.cross_norm_source(encoder_outputs)
        encdec_attention = self.cross_attention(inputs = (norm_target_selfattn, norm_encoder_outputs),
                                               mask = cross_attention_mask)
                        # 디코더의 출력(norm_target_selfattn)을 Query(query_antecedent)로,
                        # 인코더의 출력(norm_encoder_outputs)을 Key와 Value(memory_antecedent)로 사용
        # Take the residual between the output and the unnormalized target input of the cross-attention
        res_encdec_attention = encdec_attention + res_target_self_attn
                            # Residual connection of the cross-attention block

        output = self.feed_forward(res_encdec_attention)
        return output

class TransformerEncoder(nn.Module):
    """
    Stack of TransformerEncoderBlocks. Performs repeated self-attention.
    """

    def __init__(self,
                 embedding_layer, n_layers, n_heads, d_model, d_filter, dropout=None):
        super().__init__()

        self.embedding_layer = embedding_layer
        embed_size = self.embedding_layer.embed_size
        # The encoding stack is a stack of transformer encoder blocks
        self.encoding_stack = []
        for i in range(n_layers):
            encoder = TransformerEncoderBlock(embed_size, n_heads, d_filter, d_model, dropout)
            setattr(self,f"encoder{i}",encoder)
            self.encoding_stack.append(encoder)

    def forward(self, inputs, encoder_mask=None):
        """
            Args:
                inputs: Either a float32 or int32 Tensor with shape [batch_size, sequence_length, ndim]
                encoder_mask: a boolean Tensor with shape [batch_size, sequence_length, sequence_length]
            Returns:
                output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        inputs = self.embedding_layer(inputs)
        output = inputs
        for encoder in self.encoding_stack:
            output = encoder(output, self_attention_mask = encoder_mask)

        return output

class TransformerDecoder(nn.Module):
    """
        Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    def __init__(self,
                 embedding_layer, output_layer, n_layers, n_heads, d_model, d_filter, dropout = None) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        embed_size = self.embedding_layer.embed_size
        self.decoding_stack = []
        for i in range(n_layers):
            decoder = TransformerDecoderBlock(embed_size, n_heads, d_filter, d_model, dropout)
            setattr(self,f"decoder{i}",decoder)
            self.decoding_stack.append(decoder)
        self.output_layer = output_layer

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def forward(self, target_input, encoder_output, encoder_mask=None, decoder_mask=None, mask_future=False,
        shift_target_sequence_right=False):
        """
            Args:
                inputs: a tuple of (encoder_output, target_embedding)
                    encoder_output: a float32 Tensor with shape [batch_size, sequence_length, d_model]
                    target_input: either a int32 or float32 Tensor with shape [batch_size, target_length, ndims]
                    cache: Used for fast decoding, a dictionary of tf.TensorArray. None during training.
                mask_future: a boolean for whether to mask future states in target self attention

            Returns:
                a tuple of (encoder_output, output)
                    output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        if shift_target_sequence_right:
            target_input = self.shift_target_sequence_right(target_input)

        target_embedding = self.embedding_layer(target_input)
                # 대상 시퀀스의 임베딩
        
        # Build the future-mask if necessary. This is an upper-triangular mask
        # which is used to prevent the network from attending to later timesteps
        # in the target embedding
        
        # Masked Self-Attention 생성
        batch_size = target_embedding.shape[0]
        sequence_length = target_embedding.shape[1]
                # target_embedding은 대상 시퀀스의 임베딩으로 (batch_size, sequence_length, d_model(임베딩차원=hidden_size))
        self_attention_mask = self.get_self_attention_mask(batch_size, sequence_length, decoder_mask, mask_future)
                # 생성된 셀프 어텐션 마스크를 할당
                # mask_future = False면, [batch_size, sequence_length]
                # mask_future = True면, [batch_size, sequence_length, sequence_length]
        
        # Build the cross-attention mask. This is an upper-left block matrix which takes care of the masking
        # of the output shapes

        # Cross-Attention 생성
        cross_attention_mask = self.get_cross_attention_mask(
            encoder_output, target_input, encoder_mask, decoder_mask)

        # Now actually do the decoding which should take us to the right dimension
        decoder_output = target_embedding
        for decoder in self.decoding_stack:
            decoder_output = decoder(decoder_output, encoder_outputs=encoder_output, 
                                     self_attention_mask=self_attention_mask, 
                                     cross_attention_mask=cross_attention_mask)

        # Use the output layer for the final output. For example, this will map to the vocabulary
        output = self.output_layer(decoder_output)
        return output

    def shift_target_sequence_right(self, target_sequence):
        constant_values = 0 if target_sequence.dtype in [th.int32, th.int64] else 1e-10
            # 패딩된 부분을 어떻게 처리할지를 결정하는 변수. target_sequence의 형태에 따라
            # 정수 데이터의 경우 0, 아닌 경우 작은 양수
        pad_array = [1,0,0,0]
            # 패딩을 어떻게 적용할지 지정. 4차원 패딩, (batch, sequence_length, features, ...)와 같이 배치, 시퀀스 길이, 특성 및 기타 차원을 고려
            # 여기서는 시퀀스 길이를 늘리고 다른 차원은 변경하지 않는 패딩을 수행
        target_sequence = F.pad(target_sequence, pad_array, value=constant_values)[:, :-1]
            # F.pad 함수를 이용해서 target_sequence를 패딩
            # 시퀀스 길이 차원에서는 마지막 요소를 잘라내어 오른쪽으로 이동하고 나머지 차원은 변경되지 않는다
        return target_sequence

    def get_future_mask(self, batch_size, sequence_length):
        """Mask future targets and padding

            :param batch_size: a Tensor dimension
            :param sequence_length: a Tensor dimension
            :param padding_mask: None or bool Tensor with shape [batch_size, sequence_length]

            :return mask Tensor with shape [batch_size, sequence_length, sequence_length]
        """
        xind = th.arange(sequence_length)[None,:].repeat(*(sequence_length, 1))
            # [0, 1, 2, ..., sequence_length-1]의 값을 갖는 1차원 텐서
        yind = th.arange(sequence_length)[:,None].repeat(*(1, sequence_length))
            # [0]부터 [sequence_length-1]까지의 값을 가지는 열 벡터(2D 텐서)
        mask = yind >= xind
            # 각 위치 (i, j)에 대해 yind가 xind 이상인 경우 True를 가지는 마스크를 생성
            # 이로써 현재 시간 단계 이후의 위치에 대한 정보를 차단
        mask = mask[None,...].repeat(*(batch_size, 1, 1))
            # 최종 마스크의 크기는 [batch_size, sequence_length, sequence_length]
        return mask.to(get_device())

    def get_self_attention_mask(self, batch_size, sequence_length, decoder_mask, mask_future):
        # 디코더의 self-attention mask 생성. 모델이 미래 정보를 참조하지 못하게 조치.
        # decoder_mask : 현재 단계에서 어떤 토큰을 마스킹해야 하는지를 나타냄. 예를 들어, 패딩토큰은 어텐션에서 제외
        # mask_future : 미래 단계에서 어텐션을 차단할지 여부를 나타내는 불리언 값
        if not mask_future:
            return decoder_mask
            # mask_future = False인 경우 decoder_mask만 반환
            # shape : [batch_size, sequence_length]
        elif decoder_mask is None:
            return self.get_future_mask(batch_size, sequence_length)
            # mask_future=True, decoder_mask=None인 경우 미래 단계의 어텐션을 차단하는 self-attention mask 필요
            # get_future_mask 함수를 이용해서 mask_future 생성
            # shape : [batch_size, sequence_length, sequence_length]
        else:
            return decoder_mask & self.get_future_mask(batch_size, sequence_length)
            # mask_future=True, decoder_mask도 제공되는 경우 미래 mask와 decoder_mask 결합
            # shape : [batch_size, sequence_length, sequence_length]

    # This is an upper left block matrix which masks the attention for things that don't
    # exist within the internals.
    def get_cross_attention_mask(self, encoder_output, decoder_input, encoder_mask, decoder_mask):
        if encoder_mask is None and decoder_mask is None:
            cross_attention_mask = None
            # encoder_mask와 decoder_mask 모두 None 경우, 어텐션 마스크 사용 X
        
        # We need to not mask the encoding, but mask the decoding
        # The decoding mask should have shape [batch_size x target_len x target_len]
        # meaning all we have to do is pad the mask out properly
        elif encoder_mask is None:
            cross_attention_mask = decoder_mask[:, 1, :][:, None, :].repeat(
                                    *(1, encoder_output.shape[1], 1)).permute((0, 2, 1))
            # encoder_mask = None / decoder_mask 제공인 경우, 디코딩 부분만 마스킹
            # decoder_mask [batch_size, target_len] -> [batch_size, 1, target_len] 
            #                                       -> [batch_size, source_len, target_len]
            #                                       -> [batch_size, target_len, source_len]
        elif decoder_mask is None:
            cross_attention_mask = encoder_mask[:, 1, :][:, :, None].repeat(
                                    *(1, 1, decoder_input.shape[1])).permute((0, 2, 1))
            # encoder_mask 제공 / decoder_mask = None인 경우, 인코딩 부분만 마스킹
            # encoder_mask [batch_size, source_len] -> [batch_size, source_len, 1] 
            #                                       -> [batch_size, source_len, target_len]
            #                                       -> [batch_size, target_len, source_len]
        else:
            dec_attention_mask = decoder_mask[:, 1, :][:, None, :].repeat(
                                    *(1, encoder_output.shape[1], 1)).permute((0, 2, 1))
            enc_attention_mask = encoder_mask[:, 1, :][:, :, None].repeat(
                                    *(1, 1, decoder_input.shape[1])).permute((0, 2, 1))
            cross_attention_mask = th.logical_and(enc_attention_mask, dec_attention_mask)
            # encoder_maks / decoder_mask 모두 제공인 경우, 인코딩/디코딩 모두 마스킹
            # 인코더 어텐션 마스크와 디코더 어텐션 마스크 모두 True인 경우만 True
            
        # decoder_mask[:, 1, :] 및 encoder_mask[:, 1, :]
            # : 각각 디코더와 인코더 마스크에서 특정 위치에 해당하는 부분을 선택
            # 이 부분은 각각 대상 시퀀스(디코더 입력) 및 소스 시퀀스(인코더 출력)의 패딩 위치
            
        # [:, None, :], [:, :, None]
            # : 차원을 확장하여 선택된 마스크를 3D 텐서로 만듦
            # 이로써 어텐션 연산에서 이러한 마스크를 사용할 수 있게 됩니다.
            
        # repeat(*(1, encoder_output.shape[1], 1)) 및 repeat(*(1, 1, decoder_input.shape[1]))
            # : 마스크를 인코더 출력 또는 디코더 입력과 동일한 크기로 확장. 마스크를 어텐션 연산에서 사용할 수 있도록 준비
            # encoder_output은 [batch_size, sequence_length, channels]의 형태라서 '*(1, encoder_output.shape[1], 1)'은
            # 배치 차원과 임베딩 차원은 유지하고, 인코더 시퀀스 차원은 인코더 출력의 시퀀스 길이로 확장
            # decoder_input은 [batch_size, decoding_sequence_length, channels]의 형태라서 '*(1, 1, decoder_input.shape[1])'은
            # 배치 차원과 디코더 시퀀스 차원은 유지하고, 임베딩 차원을 디코더 입력의 시퀀스 길이로 확장
            
        # permute((0, 2, 1))
            # : 마스크의 차원을 재배열하여 적절한 모양을 얻습니다.
            # Transformer 어텐션에서는 일반적으로 마스크의 차원 순서를 바꾸어야 합니다.
        return cross_attention_mask

class TransformerInputEmbedding(nn.Module):

    def __init__(self,
                 embed_size,
                 vocab_size = None,
                 dropout = None,
                 batch_norm = False,
                 embedding_initializer=None) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size) # , weights=[embedding_initializer]

        self.position_encoding = PositionEmbedding(embed_size)
        self.dropout = nn.Dropout(0 if dropout is None else dropout)
        self.batch_norm = None if batch_norm is False else nn.BatchNorm1d(embed_size)

    def forward(self, inputs, start=1):

        # Compute the actual embedding of the inputs by using the embedding layer
        embedding = self.embedding(inputs)
        embedding = self.dropout(embedding)

        if self.batch_norm:
            embedding = self.batch_norm(embedding.permute((0,2,1))).permute((0,2,1))

        embedding = self.position_encoding(embedding, start=start)
        return embedding

class Transformer(nn.Module):

    def __init__(self,
                 vocab_size = None,         # 어휘 크기 (단어 집합의 크기)
                 n_layers = 6,              # Transformer 스택 내의 레이어 수
                 n_heads = 8,               # Multi-Head Self-Attention에서의 어텐션 헤드 수
                 d_model = 512,             # 모델의 임베딩 차원 및 레이어 내부의 히든 유닛 수
                 d_filter = 2048,           # 피드포워드 신경망의 필터 크기
                 dropout = None,            # 드롭아웃 비율 (선택 사항)
                 embedding_initializer=None,# 임베딩 가중치 초기화 방법 (선택 사항)
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.dropout_weight = 0 if dropout is None else dropout

        input_embedding = TransformerInputEmbedding(d_model, vocab_size, dropout) 
                    # , embedding_initializer=embedding_initializer
                    # 입력 임베딩 레이어로, 단어를 임베딩 벡터로 변환

        output_layer = EmbeddingTranspose(input_embedding.embedding)
                    # 출력 레이어로, 모델의 최종 출력을 단어로 매핑

        # Build the encoder stack.
        self.encoder = TransformerEncoder(input_embedding, n_layers, n_heads, d_model, d_filter, dropout)
                    # Transformer 인코더 스택으로, 입력 시퀀스를 인코딩

        # Build the decoder stack.
        self.decoder = TransformerDecoder(input_embedding, output_layer, n_layers, n_heads, d_model, d_filter, dropout)
                    # Transformer 디코더 스택으로, 출력 시퀀스를 디코딩

    def forward(self, source_sequence, target_sequence, encoder_mask, decoder_mask, mask_future=True, shift_target_sequence_right=True):
        # Unpack the source and target sequences from the encoder.
        # Source Sequence: [batch_size x source_length]
        # Target Sequence: [batch_size x target_length]
        #
        # Generate the masks for the encoder and decoder. There are a lot of different ways that
        # the attention masks could be passed in, so this method handles a lot of these different
        # mask shapes.
        encoder_mask = transformer_utils.convert_to_attention_mask(source_sequence, encoder_mask)
        decoder_mask = transformer_utils.convert_to_attention_mask(target_sequence, decoder_mask)

        # After the end of the encoder and decoder generation phase, we have
        # Encoder Mask: [batch_size x source_length x source_length]
        # Decoder Mask: [batch_size x target_length x target_length]

        # Next, we perform the encoding of the sentence. This should take
        # as input a tensor of shape [batch_size x source_length x input_feature_shape]
        # and generate a tensor of shape [batch_size x source_length x d_model]

        ####################################  YOUR CODE HERE  ####################################
        # PART 5: Implement the full Transformer block

        # Using the self.encoder, encode the source_sequence, and provide the encoder_mask variable as the optional mask.
        encoder_output = self.encoder(source_sequence, encoder_mask)

        # Finally, we need to do a decoding this should generate a
        # tensor of shape [batch_size x target_length x d_model]
        # from the encoder output.
        # Using the self.decoder, provide it with the decoder input, and the encoder_output. 
        
        # As usual, provide it with the encoder and decoder_masks
        # Finally, You should also pass it these two optional arguments:
        # shift_target_sequence_right = shift_target_sequence_right, mask_future = mask_future
        decoder_output = self.decoder(target_sequence, encoder_output, encoder_mask, decoder_mask, 
                                      mask_future = mask_future, shift_target_sequence_right = shift_target_sequence_right)

        return decoder_output # We return the decoder's output