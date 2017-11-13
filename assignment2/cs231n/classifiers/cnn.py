from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        #Parameters for conv layer.
        self.params['W1'] = np.random.normal(scale = weight_scale,\
                        size = (num_filters,input_dim[0],filter_size,filter_size))
        self.params['b1'] = np.zeros((num_filters))
        #Parameters for second layer:fc layer.
        #Assume stride is equal 1 and pad is half of the filter size.
        pad = (filter_size - 1) // 2
        fout_width = (input_dim[2] - filter_size + 2 * pad) + 1 
        fout_height = (input_dim[1] - filter_size + 2 * pad) + 1
        #Now compute output size after the 2x2 max pool with stride 2.
        fout_width = int((fout_width - 2) / 2 + 1)
        fout_height = int((fout_height - 2) / 2 + 1)
        #Now initialize parameters of this size.
        self.params['W2'] = np.random.normal(scale = weight_scale,\
                                size = (fout_width * fout_height * num_filters,hidden_dim))
        self.params['b2'] = np.zeros((hidden_dim))
        #Last affine layer params.
        self.params['W3'] = np.random.normal(scale = weight_scale,\
                                        size = (hidden_dim,num_classes))
        self.params['b3'] = np.zeros((num_classes))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out,conv_cache = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        #print("COnv output:",conv_out.shape)
       # print("W2",W2.shape)
        f_affine_out,f_affine_cache = affine_relu_forward(conv_out,W2,b2)
        scores,s_affine_cache = affine_forward(f_affine_out,W3,b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        #Compute softmax loss and grad.
        loss,dscores = softmax_loss(scores,y)
        #Add L2 reg.
        loss += 0.5 * self.reg * (np.sum(np.power(W1,2)) + np.sum(np.power(W2,2)) + np.sum(np.power(W3,2)))
        #Backward pass.
        d_f_affine,dw3,db3 = affine_backward(dscores,s_affine_cache)
        grads['W3'] = dw3 + self.reg * W3
        grads['b3'] = db3
        #Backward of affine-relu.
        dout,dw2,db2 = affine_relu_backward(d_f_affine,f_affine_cache)
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2
        #Backward pass of conv-relu-maxpool.
        dx,dw1,db1 = conv_relu_pool_backward(dout,conv_cache)
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
