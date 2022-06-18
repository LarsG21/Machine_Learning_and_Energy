import numpy as np
from scipy.optimize import minimize

def sigmoid(z):
    """Apply sigmoid function to z

    Args:
        z (numpy array): NxD array of real values

    Returns:
        (numpy array): NxD array with corresponding values of the sigmoid function applied to z
    """

    return 1/(1+np.exp(-z)) # <-- change this

def cost_function(w, X, y):
    """Calculate cross-entropy loss 

    Args:
        w (numpy array): Dx1 array of weights
        X (numpy array): Nx(D-1) array of inputs
        y (numpy array): Nx1 array of binary target values

    Returns:
        J (float): cross-entropy loss
        grad(numpy array): Dx1 array of gradients of cross-entropy loss
    """

    X = np.concatenate((np.ones([X.shape[0],1]), X), axis=1)  # add ones

    n = len(y)
    q = sigmoid(X@w.T)
   # print(q.shape)
    J = np.average(-y*np.log(q.T)-(1-y)*np.log(1-q.T)) # <-- change this    #log hier etspricht dem ln
    grad = (X.T@(q-y))/n# <-- change this
   # print(grad)
    return J,grad

def fit(X,y):
    """Fit logistic regression model

    Args:
        X (numpy array): Nx(D-1) array of training data
        y (numpy array): Nx1 array of target values

    Returns:
        (numpy array): Dx1 array of optimal weights
        float: loss with optimal weights
    """

    res = minimize(fun=lambda w: cost_function(w, X, y)[0], x0=np.zeros(X.shape[1]+1,), jac=lambda w: cost_function(w, X, y)[1], method='BFGS', options={'disp': True})

    return res.x, res.fun

def predict(w,X,t=.5):
    """Predict class given threshold t

    Args:
        w (numpy array): Dx1 array of weights
        X (numpy array): Nx(D-1) array of inputs
        t (float): threshold for class prediction
    Returns:
        (numpy array): Nx1 array with class prediction for every input sample
    """

    X = np.concatenate((np.ones([X.shape[0],1]), X), axis=1)  # add ones

    #a = np.round((sigmoid(X@w.T))+t-0.5)  # <-- change this
    a = np.where(sigmoid(X@w.T) > t, 1, 0)      #sigmoid um -inf bis inf auf [0,1] und dann auf- / abrunden 
    print(a)
    
    return a

def get_confusion_matrix(y_hat, y):
    """Build confusion matrix

    Args:
        y_hat (numpy array): Nx1 array of predicted classes
        y (numpy array): Nx1 array of true classes
    Returns:
        (numpy array): 2x2 array with values for confusion matrix
    """
    # ----------------------- add code here ------------------------
    i_00 = 0
    i_01 = 0
    i_10 = 0
    i_11 = 0
    
    for i in range(0,len(y)):
        if y_hat[i] == y[i] and y[i] == 1:
            i_11+=1
        if y_hat[i] == y[i] and y[i] == 0:
            i_00 +=1
        if y_hat[i] == 0 and y[i] == 1:
            i_01 +=1
        if y_hat[i] == 1 and y[i] == 0:
            i_10 +=1


    # --------------------------------------------------------------

    return np.array([[i_00, i_01],
                    [i_10, i_11]]) # <-- change this