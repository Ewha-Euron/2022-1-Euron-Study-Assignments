from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D = W.shape[0]
    C = W.shape[1]
    N = y.shape[0]
    score = np.dot(X, W)
    softmax_output = np.zeros(N)
    correct_score = score[range(N), y]
    for i in range(N) :
        class_sum = 0
        for j in range(C) :
            class_sum += np.exp(score[i,j])
        softmax_output[i] = np.exp(correct_score[i])/class_sum
        loss += -np.log(softmax_output[i])

    for i in range(N) :
        for j in range(C) :
            if y[i] == j :
                dW[:,j] += (softmax_output[i] - 1) * X[i,:]
            else :
                dW[:,j] += (np.exp(score[i,j])/np.sum(np.exp(score[i,:]))) * X[i,:]
    dW /= N  
    dW += reg * W
    loss /= N
    loss += reg * 0.5 * np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    score = np.dot(X, W)
    loss = np.exp(score[range(N), y]) / np.sum(np.exp(score), axis=1)
    loss = np.sum(-np.log(loss)) / N
    loss += reg * 0.5 * np.sum(W**2)

    dx =  np.exp(score) / np.sum(np.exp(score), axis = 1).reshape(-1, 1)
    dx[range(N), y] -= 1
    dx /= N
    dW = np.dot(X.T, dx)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
