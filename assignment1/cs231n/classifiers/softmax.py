import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i, :].dot(W)
    sum = 0
    exponents = np.zeros(num_classes)
    for j in xrange(num_classes):
      exponents[j] = np.exp(scores[j])
      sum += np.exp(scores[j])
    loss += -scores[y[i]] + np.log(sum)
    #Add regularization factor to the loss.
    loss += 0.5 * reg * np.sum(W.dot(W))
    #Divide the exponents by the sum.(helps with math later.)
    exponents /= sum
    #For the naive implementation we will loop over all classes to compute gradients.
    for k in xrange(num_classes):
        dW[:,k] += X[i,:] * exponents[k]
    #Add to the -f_yi term of the loss
    dW[:,y[i]] -= X[i,:]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  #Take mean for both loss and gradient.
  dW /= num_train
  loss /= num_train
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #Compute total scores matrix.
  scores = X.dot(W)
  #Compute the two terms for the loss.
  #First term is the neg of the sum of all the scores of the correct classes.
  f_term = -np.sum(scores[np.arange(num_train),y])
  #Second term will be log of the sum of the exponents of all the scores
  #Save computations in local variables to save in gradient computation.
  exponents = np.exp(scores)
  sums = np.sum(exponents,axis = 1)
  s_term = np.sum(np.log(sums))
  loss = (f_term + s_term) / num_train
  #End of loss computation.Compute vectorized gradient.
  scores = (exponents.transpose() / sums).transpose()
  scores[np.arange(num_train),y] -= 1
  dW += (X.T).dot(scores)
  dW /= num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

