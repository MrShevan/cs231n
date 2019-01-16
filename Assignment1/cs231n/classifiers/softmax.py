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
    
    train_num = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(train_num):
        scores = X[i].dot(W)
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(exp_scores)
        
        part_loss = exp_scores[y[i]] / sum_exp_scores
        part_loss = -np.log(part_loss)
        loss += part_loss
        
        part_dW = np.zeros_like(W)
        part_dW[:, y[i]] = -X[i]
        
        for j in range(num_classes):
            part_dW[:, j] += X[i] * (exp_scores[j] / sum_exp_scores)

        dW += part_dW
        
    loss /= train_num
    loss += reg * np.sum(W * W)
    
    dW /= train_num
    
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
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    # loss
    scores = X.dot(W) # shape 0 - number of object, shape 1 - class of object
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis = 1)
    true_class_indexes = np.arange(num_classes) == y.reshape((num_train, 1))
    correct_class_scores = exp_scores[true_class_indexes]
    loss = np.sum(-np.log(correct_class_scores / sum_exp_scores))
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # dW
    exp_scores_ = exp_scores / sum_exp_scores.reshape((sum_exp_scores.shape[0]
                                                                            ,1))
    exp_scores_[true_class_indexes] = exp_scores_[true_class_indexes] - 1
    dW = exp_scores_.T.dot(X)
    dW = dW.T
    
    dW /= num_train
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

