import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        #for t in range(W.shape[0]):
        #    dW[t,j] += X[i,t]
         #   dW[t,y[i]] -= X[i,t]
        dW[:,j] += X[i,:]
        dW[:,y[i]] -= X[i,:]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #Easy to compute gradient of multi class SVM in respect to linear classification.
  dW /= num_train

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #X dimension: (N,D)
  #W dimension:(D,C)
  scores = X.dot(W)
  #Compute vectorized Sj - Syi for each Xi(row i of X)
 # nums = np.arange(scores.shape[0])
  #scores -= np.array([[scores[nums[i],y[i]]] * scores.shape[1] for i in range(scores.shape[0])])
  correct_vec = scores[np.arange(scores.shape[0]),y]
  correct = np.ones(scores.shape) * correct_vec[:,None]
  scores -= correct
  #Add ones
  scores += np.ones(scores.shape)
  scores = np.maximum(scores,0)
  #Remember that we sum all classes except the correct class in the original loss.
  scores[scores == 1] = 0
  loss = (np.sum(scores)) / num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                               #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #Count all the elements that contribute to the loss.
  scores[scores > 0] = 1
  scores[np.arange(scores.shape[0]),y] = -1 * np.sum(scores,axis = 1)
  dW = (X.T).dot(scores)
  #Take the mean.
  dW /= num_train
  ''''#Get non zero elements of scores.
  tran_scores = scores.transpose()
  indices = np.nonzero(scores)
  for t in xrange(indices[1].shape[0]):
      dW[:,indices[1][t]] += X[indices[0][t],:]
      dW[:,y[indices[1][t]]] -= X[indices[0][t],:]
  dW /= num_train'''

  #dW[:,indices[1]] += X[indices[0]]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
