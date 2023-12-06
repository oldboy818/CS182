import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    
    # each step(each layer)에서 다음 연산을 계산
    next_h = np.tanh(np.dot(prev_h, Wh) + np.dot(x, Wx) + b)
    cache = (next_h, x, prev_h, Wx, Wh, b)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    
    next_h, x, prev_h, Wx, Wh, b = cache

    # tanh의 derivative 형태는 (1 - tanh^2)
    # dnext_h는 decoder의 upstream derivative. shape = (N, H)
    dtanh = (1 - next_h ** 2) * dnext_h

    # derivative with respect to input(x, prev_h) and param(Wx, Wh), bias(b)
    dx = np.dot(dtanh, Wx.T)    # (N,H)(H,D) = (N,D)
    dprev_h = np.dot(dtanh, Wh.T)   # (N,H)(H,H) = (N,H)
    dWx = np.dot(x.T, dtanh)    # (D,N)(N,H) = (D,H)
    dWh = np.dot(prev_h.T, dtanh)   # (H,N)(N,H) = (H,H)
    db = np.sum(dtanh, axis=0)  # (H,)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, caches = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################

    N, T, D = x.shape
    _, H = h0.shape
    h = np.zeros(shape=(N,T,H))
    caches = []

    for t in range(T):
        # 첫 번째 layer에서 hidden state 초기화
        if t == 0:
            next_h = h0
        
        # t-time step에서 forward 연산
        next_h, cache = rnn_step_forward(x[:, t, :], next_h, Wx, Wh, b)
        h[:, t, :] = next_h

        caches.append(cache)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, caches


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################

    N, T, H = dh.shape
    # next_h, x, prev_h, Wx, Wh, b = cache 이므로 x.shape = (N,D)
    _, D = cache[0][1].shape

    dx = np.zeros(shape=(N,T,D))
    dh0 = np.zeros(shape=(N,H))
    dWx = np.zeros(shape=(D,H))
    dWh = np.zeros(shape=(H,H))
    db = np.zeros(shape=(H,))

    # Upstream gradient of hidden_state at timestep t+1, shape (N, H)
    dprev_h = np.zeros_like(dh0)

    for t in reversed(range(T)):
        dnext_h = dh[:, t, :] + dprev_h
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])
        dx[:, t, :] += dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    dh0 = dprev_h
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    
    # 배열 인덱싱. 입력 단어 인덱스에 해당하는 단어 벡터를 선택. shape (N, T, D)
    # W[x]는 (V,D)에서 (N,T,D)로 확장
    out = W[x]

    cache = (x, W)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    
    x, W = cache

    # shape (V, D). V는 어휘의 크기 혹은 어휘집을 의미. D는 단어 임벡터 벡터의 차원
    dW = np.zeros_like(W)

    # x shape (N,T)이고 dout shape (N,T,D)
    np.add.at(dW, x, dout)
    
    # dW의 shape은 변하지 않으나, dW[x]는 (V,D)에서 (N,T,D)로 확장된다.
    # 결과적으로 dW의 각 행(V, 단어에 해당)에 대한 열(D, 임베딩 차원)의 값은 해당 단어에 대한
    # 그레디언트를 반영하고 업데이트된 임베딩 벡터를 나타낸다.

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    ############################################################################
    N, H = prev_c.shape

    # compute an activation vector a, shape (N, 4H)
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

    # divide an activation vecotr into four vectors, each shape (N, H)
    a_i, a_f, a_o, a_g = a[:, :H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]

    # compute next cell state c_t, shape (N, H)
    next_c = sigmoid(a_f) * prev_c + sigmoid(a_i) * np.tanh(a_g)

    # compute next hiddent state h_t, shape (N, H)
    next_h = sigmoid(a_o) * np.tanh(next_c)

    cache = (x, prev_h, prev_c, Wx, Wh, b, a_i, a_f, a_o, a_g, next_h, next_c)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################

    x, prev_h, prev_c, Wx, Wh, b, a_i, a_f, a_o, a_g, next_h, next_c = cache

    N, D = x.shape
    _, H = prev_h.shape

    # 현 셀은 비선형이 가해지지 않은 선형 상태이다. 따라서 이전 셀에 대한 그래디언트는 현재 셀의 그래디언트에
    # forget gate를 곱한 형태 (dL/dC_t * f_t).
    # 이때 현재 셀에 대한 그래디언트는 다음 셀의 그래디언트와 다음 hidden_state에 대한 그래디언트로 구성
    dnext_c += dnext_h * sigmoid(a_o) * (1 - np.tanh(next_c)**2)
    dprev_c = dnext_c * sigmoid(a_f)
    
    # compute the derivative of forget/input/output gate, shape (N,H)
    # derivative sigmoid is 'sigmoid(x) * (1 - sigmoid(x))'
    da_f = dnext_c * prev_c * (sigmoid(a_f) * (1 - sigmoid(a_f)))
    da_i = dnext_c * np.tanh(a_g) * (sigmoid(a_i) * (1 - sigmoid(a_i)))
    da_o = dnext_h * np.tanh(next_c) * (sigmoid(a_o) * (1 - sigmoid(a_o)))
    da_g = dnext_c * sigmoid(a_i) * (1 - np.tanh(a_g)**2)

    # Concatenate the gradient contributions from the gates, shape (N, 4H)
    da = np.concatenate((da_i, da_f, da_o, da_g), axis=1)

    # Compute the gradients with respect to the weights and biases
    # Wx, Wh shape (D,4H), (H,4H) / x shape (N,D) / b shape (4H,)
    dx = np.dot(da, Wx.T)   # shape (N,D)
    dprev_h = np.dot(da, Wh.T)  # shape (N,H)
    dWx = np.dot(x.T, da)   # shape (D,4H)
    dWh = np.dot(prev_h.T, da)  # shape (H,4H)
    db = np.sum(da, axis=0)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, caches = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################

    caches = []
    N, T, D = x.shape
    _, H = h0.shape

    # Initialize cell state
    C0 = np.zeros(shape=(N,H))
    # Initialize hidden state
    h = np.zeros(shape=(N,T,H))

    for t in range(T):
        if t == 0:
            prev_c = C0
            prev_h = h0
        else:
            prev_c = next_c
            prev_h = next_h
        
        # next_h shape (N,H) / next_c shape (N,H)
        next_h, next_c, cache = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
        
        # hidden state update
        h[:, t, :] = next_h
        
        # cache update
        caches.append(cache)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, caches


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################

    N, T, H = dh.shape
    _, D = cache[0][0].shape    # x shape

    dx = np.zeros(shape=(N,T,D))
    dh0 = np.zeros(shape=(N,H))
    dWx = np.zeros(shape=(D, 4*H))
    dWh = np.zeros(shape=(H, 4*H))
    db = np.zeros(shape=(4*H,))

    # Initialize graident of next Hidden_state, Cell_state
    dprev_h = np.zeros(shape=(N,H))
    dprev_c0 = np.zeros(shape=(N,H))

    for t in reversed(range(T)):
        step_dh = dh[:, t, :] + dprev_h
        step_cache = cache[t]

        if t == T-1:
            dnext_c = dprev_c0
        else:
            dnext_c = dprev_c

        dx_t, dprev_h, dprev_c, dWx_t, dWh_t, db_t = lstm_step_backward(step_dh, dnext_c, step_cache)

        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    
    dh0 = dprev_h
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    
    N, T, D = x.shape
    M = b.shape[0]

    # N개의 문장, T개의 시퀀스를 합쳐서 연산
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache

    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    # softmax 함수로 loss
    # 지수 연산의 수치 안정성을 위해 최대값을 빼준다
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    # 정규화하여 각 클래스에 대한 확률값을 구한다
    probs /= np.sum(probs, axis=1, keepdims=True)
    # Negative log-likelihood loss
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N

    # loss func에 대한 gradient를 입력 데이터에 대한 gradient로 변환하여 가중치 업데이트
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
