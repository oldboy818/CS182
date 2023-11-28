import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    x_row = x.reshape(x.shape[0], -1)
    out = np.dot(x_row, w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    x_row = x.reshape(x.shape[0], -1)
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x_row.T, dout)
    db = np.sum(dout, axis = 0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.maximum(0, x)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    dx = dout * (x > 0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        
        # compute mean and std
        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)

        # update output
        x_hat = (x - mean) / np.sqrt(var + eps)
        y = x_hat * gamma + beta

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        cache = (x, gamma, beta, eps, mean, var, x_hat, y)
        out = y

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        
        # 학습과정에서 전체 데이터 셋에 대한 평균과 분산(running_mean/var)을 활용해 추론
        out = ((x - running_mean) / np.sqrt(running_var + eps)) * gamma + beta

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    
    x, gamma, beta, eps, mean, var, x_hat, y = cache
    
    # derivative with respect to parameter(gamma, beta)
    dgamma = np.sum(dout * x_hat, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    
    # derivative with respect to input(x)
    dL_dxhat = dout * gamma
    dxhat_dx = 1. / np.sqrt(var + eps)
    dL_dvar = -0.5 * np.sum(dL_dxhat * (x - mean) * np.power(var + eps, -1.5), axis=0)
    dvar_dx = 2.0 * (x - mean) / x.shape[0]
    dL_dmean = np.sum(dL_dxhat * -1 * dxhat_dx, axis=0)
    dmean_dx = 1. / x.shape[0]
    
    dx = dL_dxhat * dxhat_dx + dL_dmean * dmean_dx + dL_dvar * dvar_dx

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    
    N, D = dout.shape
    x, gamma, beta, eps, mean, var, x_hat, y = cache

    dx_hat, m, var_sqrt = dout * gamma, len(dout), np.sqrt(var + eps)
    dx = (1. / (m * var_sqrt)) * (m * dx_hat - np.sum(dx_hat, axis=0) - x_hat * np.sum(dx_hat * x_hat, axis=0))
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        out = x * (1 - p)

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        
        dx = dout * mask
        
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C_in, H_in, W_in = x.shape
    C_out, _, H_f, W_f = w.shape

    # 패딩이 적용된 입력 데이터
    # 예를 들어 x_shape=(2,3,4,4) --> pad=1일 때, x_pad.shape=(2,3,6,6)
    x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)))  # 4-Dim zero-padding

    # Resulting dimensions. 출력의 H, W
    H_out = 1 + (H_in + 2 * pad - H_f) // stride
    W_out = 1 + (W_in + 2 * pad - W_f) // stride

    # Output
    z = np.zeros(shape=(N, C_out, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            for f in range(C_out):
                field = x_pad[:, :, i * stride : i * stride + H_f, j * stride : j * stride + W_f]
                # 입력 데이터에서 필터링할 특정 영역의 시작과 끝 인덱스를 계산
                # 입력 데이터의 i번째 행과 j번째 열에 해당하는 영역을 선택하고, 
                # 그 영역에서 필터의 높이(H_f)와 너비(W_f)에 해당하는 부분을 잘라내어 필터링을 수행
                # 컨볼루션 연산에서는 이렇게 잘라낸 부분에 필터를 적용하여 출력 피처 맵의 각 요소를 계산                

                z[:, f, i, j] = np.sum(field * w[f], axis=(-1, -2, -3)) + b[f]
                # 출력 배열 z의 각 요소는 입력 배열 x에 대한 필터 w의 컨볼루션 연산 결과
                # f는 필터의 인덱스, i와 j는 출력 배열의 위치
                # np.sum(field * w[f], axis=(-1, -2, -3)): 
                # 이 부분은 field라는 입력 배열의 일부와 필터 w[f]의 요소별(element-wise) 곱셈을 수행한 뒤, 결과 배열 전체에 대해 합을 구합니다. 
                # 여기서 axis=(-1, -2, -3)는 가장 마지막 세 차원에 대해 합을 구하라는 것을 의미
                # 이는 3차원 필터와 해당 영역의 입력 배열 간의 컨볼루션을 계산하기 위해 필요합니다.
                
                # + b[f]: 마지막으로, 필터 f에 해당하는 편향(bias)을 더합니다. 
                # 이는 컨볼루션 연산 후에 각 필터 출력에 고정된 값을 더하는 것으로, 신경망의 학습 가능한 파라미터 중 하나입니다.

    out = z

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    # 패딩이 적용된 입력 데이터
    # 예를 들어 x_shape=(2,3,4,4) --> pad=1일 때, x_pad.shape=(2,3,6,6)
    x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)))

    dx = np.zeros_like(x)
    dx_pad = np.pad(dx, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    N, C_in, H_in, W_in = x.shape
    C_out, _, H_f, W_f = w.shape

    H_out = 1 + (H_in + 2 * pad - H_f) // stride
    W_out = 1 + (W_in + 2 * pad - W_f) // stride

    for i in range(H_out):
        for j in range(W_out):
            for f in range(C_out):
                # 입력 데이터의 window
                field = x_pad[:, :, i * stride: i * stride + H_f, j * stride: j * stride + W_f]
                # Upstream derivate의 한 값으로 구성된 filter shape의 output
                # field와 같은 형태로 반환. 이 값은 해당 필터에 의해 생성된 출력 데이터의 그래디언트
                output = np.tile(dout[:, f, i, j], (*field.shape[1:], 1)).T
                # 필터 'f'에 대한 그래디언트 계산. 입력 데이터의 영역(field)와 해당 출력 그래디언트(output)의 요소별 곱을 합산하여 누적
                dw[f] += np.sum(field * output, axis=0)
                # 입력 데이터(x)에 대한 그래디언트 계산. 필터(f)와 output의 요소별 곱을 해당 위치에 누적
                dx_pad[:, :, i * stride: i * stride + H_f, j * stride: j * stride + W_f] += w[f] * output

    # 적용된 패딩을 제거
    dx = dx_pad[:, :, pad : -pad, pad : -pad]
    # 편향 그래디언트 계산
    db = np.sum(dout, axis=(0, 2, 3))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    
    H_p, W_p, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C_in, H_in, W_in = x.shape
    
    # 출력 layer의 shape
    H_out = 1 + (H_in - H_p) // stride
    W_out = 1 + (W_in - W_p) // stride
    z = np.zeros(shape = (N, C_in, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            z[:, :, i, j] = np.max(x[:, :, i*stride : i*stride + H_p,
                                          j*stride : j*stride + W_p], axis=(2,3))
    out = z
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    H_p, W_p, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    H_out = (H - H_p) // stride + 1
    W_out = (W - W_p) // stride + 1

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    # 각 윈도우(window)의 시작점을 찾기
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + H_p
                    w_end = w_start + W_p

                    # 각 윈도우에서 최대값을 갖는 원소의 위치 찾기
                    x_pooling = x[n, c, h_start:h_end, w_start:w_end]
                    max_mask = x_pooling == np.max(x_pooling)

                    # 최대값을 가진 위치에만 기울기 전달
                    dx[n, c, h_start:h_end, w_start:w_end] += dout[n, c, h, w] * max_mask

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = x.shape
    x = x.transpose(0, 3, 2, 1).reshape(-1, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 3, 2, 1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # Softmax를 구현하고, Negative log-likelihood (NLL)
    # Softmax
    # np.max(x, axis=1, keepdims=True)를 빼주는 이유는 오버플로우 방지
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))  # x차원과 동일. x에 exp 취함
    probs /= np.sum(probs, axis=1, keepdims=True)         # 행 기준 sum. 행기준 softmax
    # NLL. 평균 손실 loss 계산.
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N    # NLL 계산. 전체 평균을 loss로
    
    # Gradient 계산.
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx
