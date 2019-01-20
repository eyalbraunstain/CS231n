import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  for i in range(N):
        Xi = X[i, :]
        scores = Xi.dot(W)
        scores -= np.max(scores) # exp norm factor
        sum_j = np.sum(np.exp(scores))
        p = lambda k: np.exp(scores[k]) / sum_j             
        Li = -np.log(p(y[i]))
        loss += Li
        
        for k in range(C):
            dW[:, k] += (p(k) - (y[i] == k)) * X[i]            
            
  loss /= N
  loss += reg * np.sum(W * W)
  
  dW /= N
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################  
  N = X.shape[0]
  scores = X.dot(W) # shape (N, C)
  scores -= np.max(scores, axis=1)[:, np.newaxis] # exp norm factor
  s_exp = np.exp(scores);
  s_exp_sum = np.sum(s_exp, axis=1)
  probs = s_exp / s_exp_sum[:, np.newaxis] # shape (N, C)
  probs_y = probs[np.arange(N), y] # shape (N, 1)
  losses = -np.log(probs_y) # shape (N, C)
  loss += np.sum(losses)
  loss /= N
  loss += reg * np.sum(W * W)
    
  ind_y = np.zeros_like(probs) # shape (N, C)
  ind_y[np.arange(N), y] = 1
  dW = X.T.dot(probs - ind_y)
  dW /= N
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

