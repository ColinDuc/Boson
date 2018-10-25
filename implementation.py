#### FUNCTIONS ###

# Useful starting lines
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#%load_ext autoreload
#%autoreload 2

from proj1_helpers import *
from utilities import *


### GRADIENT DESCENT ###

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    #ws = [initial_w]
    #losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        #loss = compute_mse(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        """# store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))"""

    return w#,loss


### STOCHASTIC GRADIENT DESCENT ###

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    #ws = [initial_w]
    #losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            #loss = compute_mse(y, tx, w)
            # store w and loss
            #ws.append(w)
            #losses.append(loss)
        #loss=compute_mse(y, tx, w)
       # print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))"""
    return w


### LEAST SQUARES ####

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


### RIDGE REGRESSION ###

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


### LOGISTIC REGRESSION ###

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """An iterative method (gradient descent) for calculating an optimal w for the cost function of the logistic regression approach"""
    #Ensure that the dimensions of the inputs are the one we need
    y = y.ravel()
    if(tx.ndim == 1):
        tx = tx.reshape((-1,1))
    w = initial_w.ravel()
 
    for n_iter in range(max_iters):
        w=w-gamma*logistic_grad(y, tx, w) #modify w by the gradient with respect to w of the associated loss function.
        
        #Prints the loss every 100 iterations, as well as at the end
        if(n_iter == max_iters-1):
               print('n = {}, loss = {}'.format(n_iter,logistic_loss(y,tx,w)), end = "\n")
        elif (n_iter % 100 == 0):
               print('n = {}, loss = {}'.format(n_iter,logistic_loss(y,tx,w)), end = "\r")

    return (w,logistic_loss(y,tx,w))


### REGULARIZED LOGISTIC REGRESSION ###

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_) :
    """An iterative method (gradient descent) for calculating an optimal w for the cost function of the regularized logistic regression approach"""
    #Ensure that the dimensions of the inputs are the one we need
    y = y.ravel()
    if(tx.ndim == 1):
        tx = tx.reshape((-1,1))
    w = initial_w.ravel()

    for n_iter in range(max_iters):
        # While updating w, gamma is modified to take into account the size of the sample.
        w=w-gamma*reg_logistic_grad(y, tx, w, lambda_)
    return (w,logistic_loss(y,tx,w)) # calculate the actual loss using the logistic_loss function, as it is this quantity that has intrinsic meaning

