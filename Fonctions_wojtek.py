import numpy as np
from proj1_helpers import *

""" --------------------------------------------------- """
""" ------------ Required implementations ------------- """
""" --------------------------------------------------- """

#GRADIENT DESCENT WITH LEAST SQUARES LOSS
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Looks for w using an iterative gradient descent algorithm. At each step, the gradient of the mean-square-error with respect to w is calculated with compute_grad_mse"""
    
    #Ensure that the dimensions of the inputs are the one we need
    y = y.ravel()
    if(tx.ndim == 1):
        tx = tx.reshape((-1,1))
    w = initial_w.ravel()
    
    for n_iter in range(max_iters):
        w=w-gamma*compute_grad_mse(y,tx,w) #at each iteration, perform a gradient descent, modulated by gamma

    return (w,compute_mse(y,tx,w))


#STOCHASTIC GRADIENT DESCENT WITH LEAST SQUARES LOSS
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Looks for w using an iterative stochastic gradient descent algorithm. Works faster than least_squares_SGD, as it chooses a sample and using it, performs one descent (one iteration)"""
    
    #Ensure that the dimensions of the inputs are the one we need
    y = y.ravel()
    if(tx.ndim == 1):
        tx = tx.reshape((-1,1))
    w = initial_w.ravel()
    
    nOfSamples=np.shape(y)[0]
    for n_iter in range(max_iters):
        rn=np.random.randint(0,nOfSamples)
        grad=-np.dot(np.transpose(tx[rn,:]),(y[rn]-np.dot(tx[rn,:],w.ravel()))) # we are not calling compute_grad_mse as it works only on more than one sample
        w=w-gamma*grad   

    return (w,compute_mse(y,tx,w))


#LEAST SQUARES : EXACT RESOLUTION
def least_squares(y, tx):
    """Calculate the least squares solution (minimizes the squared error (y-tx*w)^2, using the normal equations"""
    #Ensure that the dimensions of the inputs are the one we need
    y = y.ravel()
    if(tx.ndim == 1):
        tx = tx.reshape((-1,1))

    w_star=np.linalg.solve((tx.T).dot(tx), (tx.T).dot(y)) #computes the solution analytically, by solving a system of linear equations
    return (w_star, compute_mse(y,tx,w_star))

#RIDGE REGRESSION WITH LEAST SQUARES AND NORM-2 REGULARIZATION
def ridge_regression(y, tx, lambda_):
    """Ridge regression with normal equations. Algorithm analogous to least-squares, but the minimized problem is now (y-tx*w)^2 + lambda*norm(w)^2"""
    #Ensure that the dimensions of the inputs are the one we need
    y = y.ravel()
    if(tx.ndim == 1):
        tx = tx.reshape((-1,1))

    N=tx.shape[0]
    D=tx.shape[1]
    w_star = np.linalg.solve(np.dot(tx.T,tx) + 2*N*lambda_*np.eye(D), (tx.T).dot(y)) # the lambda is scaled by the N factor to take into account the number of features
    return (w_star, compute_mse(y,tx,w_star))

#GRADIENT DESCENT WITH LOGISTIC REGRESSION
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

#REGULARIZED LOGISTIC REGRESSION
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

""" --------------------------------------------------- """
""" ----------- Cost and utility functions ------------ """
""" --------------------------------------------------- """
#LEAST SQUARES LOSS
def compute_mse(y, tx, w):
    """Computes the Mean-Square-Error"""
    error = y-np.dot(tx,w) #precompute the error to avoid using it computing it twice
    return np.dot(np.transpose(error),error)/(2*y.shape[0])

#LEAST SQUARES GRADIENT
def compute_grad_mse(y, tx, w):
    """Computes the gradient, with respect to w, of the Mean-Square-Error loss function"""

    return -np.dot(np.transpose(tx),(y-np.dot(tx,w)))/(y.shape[0])

#SIGMOID FUNCTION
def sigmoid(t):
    """apply sigmoid function on t."""
    
    """ The positive and negative values of e are treated
    separately to avoid overflows."""
    
    neg_ind=(t < 0)
    pos_ind=(t > 0)
    sig=np.zeros(t.shape)
    
    sig[neg_ind]=np.exp(t[neg_ind])/(1+np.exp(t[neg_ind]))
    sig[pos_ind]=1/(1+np.exp(-t[pos_ind]))
    return sig

#LOGISTIC REGRESSION LOSS
def logistic_loss(y, tx, w):
    """Returns the loss associated calculated for the cost function: -log_likelihood(the flags y were produced given the samples tx and the weights w)"""
    # ***************************************************
    e=tx.dot(w)
    
    """ The positive and negative values of e are treated
    separately to avoid overflows."""
    log_term = np.log(1 + np.exp(-np.absolute(e))) + np.maximum(0, e)
    return np.sum(log_term - y * e)
#LOGISTIC REGRESSION GRADIENT
def logistic_grad(y, tx, w):
    """return the gradient"""
    e=tx.dot(w)
    sig_e=sigmoid(e)  
    return (tx.T).dot(sig_e-y)

#REGULARIZED LOGISTIC REGRESSION GRADIENT
def reg_logistic_grad(y, tx, w, lambda_):
    """return the gradient for the regularized logistic regression"""
    e=tx.dot(w)
    sig_e=sigmoid(e)

    return (tx.T).dot(sig_e-y)+lambda_*w

#SPLIT DATA IN 2"

def split_data_test(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

