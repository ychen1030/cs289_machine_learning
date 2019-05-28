from builtins import range
from builtins import object
import numpy as np

from layers import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of [H, ...], and perform classification over C classes.

    The architecure should be like affine - relu - affine - softmax for a one
    hidden layer network, and affine - relu - affine - relu- affine - softmax for
    a two hidden layer network, etc.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim, hidden_dim=[10, 5], num_classes=8,
                 weight_scale=0.1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        self.hidden_dim = hidden_dim

        self.num_layers = 1 + len(hidden_dim)
        ############################################################################
        # TODO: Initialize the weights and biases of the net. Weights              #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        for i in range(len(hidden_dim)):
            sub = str(i + 1)
            if i == 0:
                self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim[0])
            else:
                self.params['W' + sub] = weight_scale * np.random.randn(hidden_dim[i - 1], hidden_dim[i])
            self.params['b' + sub] = np.zeros(hidden_dim[i])

        self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(hidden_dim[-1], num_classes)
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
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
        # TODO: Implement the forward pass for the net, computing the              #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        caches = {}
        norm_cache, drop_cache = [], []
        out = X
        last_w, last_b = 'W' + str(self.num_layers), 'b' + str(self.num_layers)
        for i in range(1, self.num_layers):
            a, fc_cache = affine_forward(out, self.params['W' + str(i)], self.params['b' + str(i)])
            out, relu_cache = relu_forward(a)
            caches[i] = (fc_cache, relu_cache, norm_cache, drop_cache)

        scores, fc_cache = affine_forward(out, self.params[last_w], self.params[last_b])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the net. Store the loss            #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k].                                                          #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        dx, grads[last_w], grads[last_b] = affine_backward(dx, fc_cache)

        for i in range(self.num_layers - 1, 0, -1):
            fc_cache, relu_cache, norm_cache, drop_cache = caches[i]
            da = relu_backward(dx, relu_cache)
            dx, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(da, fc_cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
