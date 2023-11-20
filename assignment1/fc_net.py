import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *

from deeplearning import layer_utils
from deeplearning import layers


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        
        # W1.shape = (D, H) = (input_dim, hidden_dim) = (5, 50)
        # W2.shape = (H, C) = (hidden_dim, num_classes) = (50, 7)
        self.params['W1'] = np.random.normal(scale = weight_scale, size = (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
        
        # b1.shape = (H, 1) = (hidden_dim, 1) = (50, 1)
        # b2.shape = (C, 1) = (num_classes, 1) = (7, 1)
        self.params['b1'] = np.zeros(shape = (hidden_dim, ))
        self.params['b2'] = np.zeros(shape = (num_classes, ))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        a1, fc1_ReLU_cache = layer_utils.affine_relu_forward(x = X, w = self.params['W1'], b = self.params['b1'])
        scores, fc2_cache = layers.affine_forward(x = a1, w = self.params['W2'], b = self.params['b2'])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization on the weights,    #
        # but not the biases.                                                      #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # dL/dz2. 상류에서 전달된 기울기
        loss, delta = layers.softmax_loss(x = scores, y = y)
        # FC layer (affine layer) 역전파 계산
        delta, grads['W2'], grads['b2'] = layers.affine_backward(dout = delta, cache = fc2_cache)
        # Activation layer 역전파 계산
        delta, grads['W1'], grads['b1'] = affine_relu_backward(dout = delta, cache = fc1_ReLU_cache)

        # L2 Regularization 적용
        W1, W2 = self.params['W1'], self.params['W2']
        penalty = 0.5 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        loss += self.reg * penalty
        # regularization이 적용된 gradient update
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        # W1.shape = (D, H1) = (input_dim, hidden_dim[0]) = (15, 20)
        # W2.shape = (H1, H2) = (hidden_dim[0], hidden_dim[1]) = (20, 30)
        # W3.shape = (H2, C) = (hidden_dim[1], num_classes) = (30, 10)
        self.params['W1'] = np.random.normal(scale = weight_scale, size = (input_dim, hidden_dims[0]))
        self.params['W2'] = np.random.normal(scale = weight_scale, size = (hidden_dims[0], hidden_dims[1]))
        self.params['W3'] = np.random.normal(scale = weight_scale, size = (hidden_dims[1], num_classes))

        # b1.shape = (H1, ) = (hidden_dims[0], ) = (20, )
        # b2.shape = (H2, ) = (hidden_dims[1], ) = (30, )
        # b2.shape = (C, ) = (num_classes, ) = (10, )
        self.params['b1'] = np.zeros(shape = (hidden_dims[0], ))
        self.params['b2'] = np.zeros(shape = (hidden_dims[1], ))
        self.params['b3'] = np.zeros(shape = (num_classes, ))
        
        if use_batchnorm:
            self.params['gamma1'] = np.ones(shape = (hidden_dims[0], ))
            self.params['gamma2'] = np.ones(shape = (hidden_dims[1], ))

            self.params['beta1'] = np.zeros(shape = (hidden_dims[0], ))
            self.params['beta2'] = np.zeros(shape = (hidden_dims[1], ))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        # First layer
        a1, fc1_cache = layers.affine_forward(x = X, w = self.params['W1'], b = self.params['b1'])
        if self.use_batchnorm:
            a1, fc1_bn_cache = layers.batchnorm_forward(x=a1, gamma=self.params['gamma1'],
                                                    beta=self.params['beta1'], bn_param=self.bn_params[0])
        a1, fc1_relu_cache = layers.relu_forward(x=a1)
        if self.use_dropout:
            a1, fc1_drop_cache = layers.dropout_forward(x=a1, dropout_param=self.dropout_param[0])
        
        # Second layer
        a2, fc2_cache = layers.affine_forward(x = a1, w = self.params['W2'], b = self.params['b2'])
        if self.use_batchnorm:
            a2, fc2_bn_cache = layers.batchnorm_forward(x=a2, gamma=self.params['gamma2'],
                                                    beta=self.params['beta2'], bn_param=self.bn_params[1])
        a2, fc2_relu_cache = layers.relu_forward(x=a2)
        if self.use_dropout:
            a2, fc2_drop_cache = layers.dropout_forward(x=a2, dropout_param=self.dropout_param[1])
        
        # FC layer
        z3, fc3_cache = layers.affine_forward(x=a2, w=self.params['W3'], b=self.params['b3'])

        scores = z3
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization on the         #
        # weights, but not the biases.                                             #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # Third layer backpropagation (Output layer)
        loss, delta = layers.softmax_loss(x=scores, y=y)
        delta, grads['W3'], grads['b3'] = layers.affine_backward(dout=delta, cache=fc3_cache)

        # Second layer backpropagation
        if self.use_dropout:
            delta = layers.dropout_backward(dout=delta, cache=fc2_drop_cache)
        delta = layers.relu_backward(dout=delta, cache=fc2_relu_cache)
        if self.use_batchnorm:
            delta, grads['gamma2'], grads['beta2'] = layers.batchnorm_backward(dout=delta, cache=fc2_bn_cache)
        delta, grads['W2'], grads['b2'] = layers.affine_backward(dout=delta, cache=fc2_cache)
        
        # First layer backpropagation
        if self.use_dropout:
            delta = layers.dropout_backward(dout=delta, cache=fc1_drop_cache)
        delta = layers.relu_backward(dout=delta, cache=fc1_relu_cache)
        if self.use_batchnorm:
            delta, grads['gamma1'], grads['beta1'] = layers.batchnorm_backward(dout=delta, cache=fc1_bn_cache)
        delta, grads['W1'], grads['b1'] = layers.affine_backward(dout=delta, cache=fc1_cache)

        # L2 Regularization 적용
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        penalty = 0.5 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        loss += self.reg * penalty
        
        # regularization이 적용된 gradient update
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


if __name__ == '__main__':
    pass