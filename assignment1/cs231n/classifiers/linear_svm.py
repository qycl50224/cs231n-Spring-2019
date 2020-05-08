from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    '''
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
     '''
    

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW = np.zeros(W.shape) # initialize the gradient as zero
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            # 这里margin代表了第i个训练样本分类为j的程度（j≠i）也就是分类错误的程度，
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            # margin大于0可以理解为错误程度我们已经无法接受
            if margin > 0:
                # 下面的代码是为了计算总loss
                loss += margin
                # 下面是算导数，如果是分类错误，让导数大一点，也就会下降快一些
                # 注意这里是切片索引
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    dW /= num_train
    dW += reg * W
    loss /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    correct_scores = scores[np.arange(num_train), y].reshape(num_train,-1) # reshape -1 意味着这个维度的长度由给定的其他维度长度来计算
    # 下面通过广播来计算margins
    margins = np.maximum(0, scores - correct_scores + 1)
    # 由于上面的margin计算时把j==i也就是分类正确的错误程度也划分了1，我们下面把它改为0
    # 下面为nupmy花式索引
    margins[np.arange(num_train), y] = 0
    # 加上正则项
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)
    margins[margins > 0] = 1
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] -= row_sum
    
    dW += np.dot(X.T, margins)/num_train + 2 * reg * W  

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
