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

    scores = X.dot(W) # dot 연산을 통해 score를 얻음. score의 shape은 (C,)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Softmax Loss
    for i in range(num_train): # 모든 사진에 대해 반복
        f = scores[i]
        softmax = np.exp(f)/np.sum(np.exp(f)) # score를 확률적으로 변환하여 softmax에 저장
        loss += -np.log(softmax[y[i]])  # class에 대한 확률 값 중 정답인 class의 확률 값만을 -log를 취해 loss 함수에 더함
        
        # Weight Gradient
        for j in range(num_classes):
            # 모든 class를 돌며 image pixel과 그 softmax 값을 곱한 값을 dW로 업데이트
            dW[:,j] += X[i] * softmax[j] 
        dW[:,y[i]] -= X[i] # 정답 클래스는 한 번만 image pixel value 만큼 빼줌
    
    # Average
    loss /= num_train 
    dW /= num_train

    # Overfitting을 방지하기 위해 L2 Regularization
    loss += reg * np.sum(W * W) 
    dW += reg * 2 * W

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

    num_train = X.shape[0]
    scores = X.dot(W) 

    # naive 버전에서는 for loop를 써서 속도상의 문제가 생김
    # loop를 쓰지 않고 한번에 vector 연산
    
    # Softmax Loss
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    softmax_matrix = np.exp(scores)/sum_exp_scores
    loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]))

    # Weight Gradient
    softmax_matrix[np.arange(num_train), y] -= 1
    dW = X.T.dot(softmax_matrix)

    # Average
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW