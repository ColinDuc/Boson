import numpy as np
import matplotlib.pyplot as plt

# Utility Functions


### Loss functions ###

def calculate_mse(e):
    """Calculate mse for vector e"""
    return 1/2*np.mean(e**2)

def compute_mse(y, tx, w):
    """compute the loss by mse"""
    error = y - tx.dot(w)
    mse = error.dot(error) / (2 * len(error))
    return mse

def calculate_mae(e):
    """Calculate the mae for vector e"""
    return np.mean(np.abs(e))


### Utilities for GD & SGD ###

def compute_gradient(y, tx, w):
    """Compute the gradient"""
    error = y - tx.dot(w)
    gradient = -tx.T.dot(error) / len(error)
    return gradient, error


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels"""
    error = y - tx.dot(w)
    gradient = -tx.T.dot(error) / len(error)
    return gradient, error


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Useful for SGD
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
            
### Polynomial ###

def build_poly(x, degree):
    """ create polynomial basis functions for data x, for j = 0 up to j = degree"""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

        
### Split Data ###

def split_data(x, y, ratio, seed=1):
    """split the dataset in two parts (training & test) based on the split ratio"""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    
    # create split, we obtain a training set and a test set
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def split_data_K(x, y, K, seed=1):
    """split the dataset in K parts, in egal parts"""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(num_row/K))
    x_K = []
    y_K = []
    for i in range(K):
        if i!=K:
            index = indices[: i*index_split]
        else:
            index = indices[i*index_split:]
        # create split
        x_K.append(x[index])
        y_K.append(y[index])
    return np.asarray(x_K), np.asarray(y_K)


def test_select(x_K,y_K,n):
    """ Choose one part to be the test set, the K-1 are the training set"""
    if n>len(x_K):
        print("n is greater than the length of x_K")
        return
    x_train=x_K
    y_train=y_K
    x_te=x_K[n]
    y_te=y_K[n]
    x_train=np.delete(x_train,n)
    y_train=np.delete(y_train,n)
    return x_train,y_train,x_te,y_te
  
    
def build_k_indices(y, K, seed):
    """build k array of indices of y"""
    num_row = y.shape[0]
    interval = int(num_row / K)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    K_indices = [indices[k * interval: (k + 1) * interval] for k in range(K)]
    return np.array(K_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression"""
    # set k is test set, the others are the training set
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    # form data with polynomial degree, using build_poly
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    
    # perform ridge regression
    w = ridge_regression(y_tr, tx_tr, lambda_)
    
    # calculate the loss for train and test data   A NOTER: ici, on calcule sqrt(2*mse) POURQUOI ?
    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
    return loss_tr, loss_te,w

#def cross_validation_ALAIN(y, x, k_indices,


### Results ###
    
def definitive_res(x):      
    """from the regression, need to choose if we assign +1 or -1. We assign -1 if the value is <0 and +1 if it is >=0"""
    x[x<0] = -1
    x[x>=0] = 1
    return x


def definitive_res_logistic(x, threshold):
    """from the logistic regression, need to choose if we assign +1 or -1. But here we have a probability, thus we need a different classification than before"""
    for i in range(len(x)):
        if x[i] < threshold:
            x[i] = -1
        else:
            x[i] = 1
    return x
          

### Matrix standardization ###

def standardize(x):
    """Standardization"""
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data


### SIGMOID FUNCTION  ###

def sigmoid(t):
    """apply sigmoid function on t"""
    
    """ The positive and negative values of e are treated
    separately to avoid overflows"""
    neg_ind=(t < 0)
    pos_ind=(t > 0)
    sig=np.zeros(t.shape)
    
    sig[neg_ind]=np.exp(t[neg_ind])/(1+np.exp(t[neg_ind]))
    sig[pos_ind]=1/(1+np.exp(-t[pos_ind]))
    return sig


### LOGISTIC REGRESSION LOSS ###

def logistic_loss(y, tx, w):
    """Returns the loss associated calculated for the cost function: -log_likelihood"""
    e=tx.dot(w)
    
    """ The positive and negative values of e are treated
    separately to avoid overflows."""
    log_term = np.log(1 + np.exp(-np.absolute(e))) + np.maximum(0, e)
    return np.sum(log_term - y * e)

### LOGISTIC REGRESSION GRADIENT ###

def logistic_grad(y, tx, w):
    """return the gradient for the logistic regression"""
    e=tx.dot(w)
    sig_e=sigmoid(e)  
    return (tx.T).dot(sig_e-y)


### REGULARIZED LOGISTIC REGRESSION GRADIENT ###

def reg_logistic_grad(y, tx, w, lambda_):
    """return the gradient for the regularized logistic regression"""
    e=tx.dot(w)
    sig_e=sigmoid(e)
    return (tx.T).dot(sig_e-y)+lambda_*w


### INTERACTION BETWEEN FEATURES ###     


                                       #     """RAYAN !!!!!!!!"""

def interaction_prod(x,k=0,square=True):
    if k>len(x[0]) or k==0:
        k=len(x[0])
    if square==True:
        a=0
    else:
        a=1
    for i in range(a,k):
        if square:
            for j in range(i+1):
                print(i,j)
                x = np.c_[x, x[:, i] * x[:, j]]
        else:
            for j in range(i):
                print(i,j)
                x = np.c_[x, x[:, i] * x[:, j]]
    return x

def interaction_prodbis(x,k=0,square=True):
    if k>len(x[0]) or k==0:
        k=len(x[0])
    if square==True:
        a=0
    else:
        a=1   
    for i in range(a,k):
        if square:
            for j in range(i+1):
                print(i,j)
                z = np.c_[z, x[:, i] * x[:, j]]
        else:
            for j in range(i):
                print(i,j)
                z = np.c_[z, x[:, i] * x[:, j]]
    return z

##### CLASSIFICATION FROM JET_NUM ####

def one_hot_jet_num(x,indx):
    x = np.concatenate((x, np.zeros((len(x),4))), axis = 1)
    for i in range(4):
        x[:,-(i+1)] = (x[:,indx] == i).astype(float)
    return np.delete(x,indx,1)
