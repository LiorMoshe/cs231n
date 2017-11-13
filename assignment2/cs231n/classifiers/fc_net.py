from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
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
        self.params['b1'] = np.zeros((hidden_dim))
        self.params['b2'] = np.zeros((num_classes))
        self.params['W1'] = np.random.normal(scale = weight_scale,size = (input_dim,hidden_dim))
        self.params['W2'] = np.random.normal(scale = weight_scale,size = (hidden_dim,num_classes))
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
        #Use forward passes that we wrote.
        fOut,fCache = affine_relu_forward(X,self.params['W1'],self.params['b1'])
        scores,sCache = affine_forward(fOut,self.params['W2'],self.params['b2'])
       # relu_node = np.maximum(X.dot(self.params['W1']) + b1,0)
        #scores = relu_node.dot(self.params['W2']) \
        #             + self.params['b2']

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
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #Use softmax function(we already wrote a similar function in assignment1)
        loss,dscores = softmax_loss(scores,y)
        #Add L2 regularization to the loss.
        loss += 0.5 * self.reg * (np.sum(np.power(self.params['W1'],2)) + \
                        np.sum(np.power(self.params['W2'],2)))
        #Use backward pass that we wrote.
        dRelu,dW2,db2 = affine_backward(dscores,sCache)
        grads['W2'] = dW2 + self.reg * self.params['W2']
        grads['b2'] = db2
        dx,dW1,db1 = affine_relu_backward(dRelu,fCache)
        grads['W1'] = dW1 + self.reg * self.params['W1']
        grads['b1'] = db1
        #Compute gradients using backward pass that we wrote.
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_batchnorm_relu_forward(x,w,b,gamma,beta,bn_param,dropout = None):
    '''Helper function that does forward pass for affine-[batch norm]-relu layers at
     once.
     Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
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
    '''
    #Run forward pass for affine layer.
    fc_out,fc_cache = affine_forward(x,w,b)
    #Run forward pass for batch_norm
    batchnorm_out,batchnorm_cache = batchnorm_forward(fc_out,gamma,beta,bn_param)
    #Run forward pass for relu.
    out,relu_cache = relu_forward(batchnorm_out)
    #In case of a dropout.
    dropout_cache = None
    if (dropout != None):
        out,dropout_cache = dropout_forward(out,dropout)
    cache = (fc_cache,batchnorm_cache,relu_cache,dropout_cache)
    return out,cache


def affine_batchnorm_relu_backward(dout,cache,dropout = None):
    #Extract variables from cache.
    fc_cache,batchnorm_cache,relu_cache,dropout_cache = cache
    #In case of a dropout.
    if (dropout != None):
        dout = dropout_backward(dout,dropout_cache)
    #Run backward pass for relu.
    da = relu_backward(dout,relu_cache)
    #Run backward pass for batchnorm.
    dy,dgamma,dbeta = batchnorm_backward_alt(da,batchnorm_cache)
    #Run backward pass for affine layer.
    dx,dw,db = affine_backward(dy,fc_cache)
    return dx,dw,db,dgamma,dbeta


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

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
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
        #First layer
        self.params['W1'] = np.random.normal(scale = weight_scale,size = (input_dim,hidden_dims[0]))
        self.params['b1'] = np.zeros((hidden_dims[0]))
        if(self.use_batchnorm):
            self.params['gamma1'] = np.ones((hidden_dims[0]))
            self.params['beta1'] = np.zeros((hidden_dims[0]))
        #Hidden layers.

        for i in range(0,self.num_layers - 2):
            self.params['W' +   str(i + 2)] = np.random.normal(scale = weight_scale,
                                                    size = (hidden_dims[i],hidden_dims[i + 1]))
            self.params['b' + str(i + 2)] = np.zeros((hidden_dims[i + 1]))
            #Add gamma and beta parameters for batch normalization.
            if (self.use_batchnorm):
                self.params['gamma' + str(i + 2)] = np.ones((hidden_dims[i + 1]))
                self.params['beta' + str(i + 2)] = np.zeros((hidden_dims[i + 1]))
        #Last layer.
        #Add edge case where there is only one hidden layer.(a bit messy)
        if (self.num_layers == 2):
            i = 0
        else:
            i += 1
        self.params['W' + str(i + 2)] = np.random.normal(scale = weight_scale,
                                                        size = (hidden_dims[i],num_classes))
        self.params['b' + str(i + 2)] = np.zeros((num_classes))
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
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        caches = []
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
        #Each layer is followed with a relu nonlinear activation besides the last layer.
        curr_out = X
        #All layers besides the last layer are defined as affine-relu
        dropout_dict = None
        if (self.use_dropout):
            dropout_dict = self.dropout_param
        for i in range(self.num_layers - 1):
           #Seperate cases when we use batchnorm and when we don't use it.
           if (self.use_batchnorm):
               next_out,curr_cache = affine_batchnorm_relu_forward(curr_out,
                                                         self.params['W' + str(i + 1)],
                                                        self.params['b' + str(i + 1)],
                                                        self.params['gamma' + str(i + 1)],
                                                        self.params['beta' + str(i + 1)],
                                                        self.bn_params[i],dropout_dict)

           else:
            next_out,curr_cache = affine_relu_forward(curr_out,
                                                        self.params['W' + str(i + 1)],
                                                        self.params['b' + str(i + 1)],
                                                        dropout_dict)
           #Add the cache.
           caches = caches + [curr_cache]
           #Save the output.
           curr_out = next_out
        #Last layer.
        i += 1
        scores,last_cache = affine_forward(curr_out,self.params['W' + str(i + 1)],
                                            self.params['b' + str(i + 1)])
        caches = caches + [last_cache]
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
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #Compute softmax loss with given function.Add L2 regularization.
        loss,dscores = softmax_loss(scores,y)
        for i in range(self.num_layers):
            loss += 0.5 * self.reg * (np.sum(np.power(self.params['W' + str(i + 1)],2)))
        #Run the backward pass using the cache that we saved.
        #Last layer doesn't use relu nonlinear activation.
        curr_dx,curr_dw,curr_db = affine_backward(dscores,caches[len(caches) - 1])
        grads['W' + str(self.num_layers)] = curr_dw + self.reg * self.params[
                                                        'W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = curr_db
        #Backward pass over all the affine-relu layers.
        for i in range(len(caches) - 2,-1,-1):
            if (self.use_batchnorm):
                curr_dx,dw,db,dgamma,dbeta = \
                                    affine_batchnorm_relu_backward(curr_dx,caches[i],dropout_dict)
            else:
                curr_dx,dw,db = affine_relu_backward(curr_dx,caches[i],dropout_dict)
            grads['W' + str(i + 1)] = dw + self.reg * self.params[
                                            'W' + str(i + 1)]
            grads['b' + str(i + 1)] = db
            #If we use batch normalization add gamma and beta gradients.
            if (self.use_batchnorm):
                grads['gamma' + str(i + 1)] = dgamma
                grads['beta' + str(i + 1)] = dbeta
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
