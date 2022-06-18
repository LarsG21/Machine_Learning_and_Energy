import numpy as np
import LinearRegression as lr

def normalize_features(Phi_test, Phi_train):
    """Normalizes features except first column (constant).

    Args:
        Phi_test (np.array): N x M design matrix for test data
        Phi_train (np.array): N x M design matrix for training data

    Returns:
        (np.array, np.array): Normalized design matrices
    """
    mu = np.mean(Phi_train, axis=0)
    mu[0] = 0
    sigma = np.std(Phi_train, axis=0)
    sigma[0] = 1
    return (Phi_test-mu)/sigma, (Phi_train-mu)/sigma



def model_1(X_test, X_train, y_train, lam=1e-10, return_weights=False):
    """Trains linear regression model and returns predictions for test data.
        Model is given by: y = w_0 + w_1*x

    Args:
        X_test (numpy array): N x 1 array of test inputs
        X_train (numpy array): N x 1 array of training inputs
        y_train (numpy array): N x 1 array of training targets
        lam (float, optional): Controls strength of regularization.. Defaults to 1e-10.
        return_weights (bool, optional): If True function also return model weights. Defaults to False.

    Returns:
        (np.array, np.array(optional)): Test set predictions and weights (if return_weights = True)
    """
    Phi_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis=1)
                                    #einsen wie shape Xtrain[0]
    Phi_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis=1)

    Phi_test, Phi_train = normalize_features(Phi_test, Phi_train)
    
    w = lr.fit(Phi_train, y_train, lam)
    y_pred = lr.predict(Phi_test, w)

    if return_weights is True:
        return y_pred, w
    if return_weights is False:
        return y_pred



def model_2(X_test, X_train, y_train, lam=1e-10, return_weights=False):
    """Trains linear regression model and returns predictions for test data.
        Model is given by: y = w_0 + w_1*x + w_2*x^2 + w_3*sin(1.5*x)
    
    Args:
        X_test (numpy array): N x 1 array of test inputs
        X_train (numpy array): N x 1 array of training inputs
        y_train (numpy array): N x 1 array of training targets
        lam (float, optional): Controls strength of regularization.. Defaults to 1e-10.
        return_weights (bool, optional): If True function also return model weights. Defaults to False.

    Returns:
        (np.array, np.array(optional)): Test set predictions and weights (if return_weights = True)
    """
    # ------------ add code here -----------------------------------
    Phi_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train, X_train**2, np.sin(1.5*X_train)), axis=1) # <-- change this
    Phi_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test, X_test**2, np.sin(1.5*X_test)), axis=1) # <-- change this

    Phi_test, Phi_train = normalize_features(Phi_test, Phi_train)
    
    w = lr.fit(Phi_train, y_train, lam)
    y_pred = lr.predict(Phi_test, w) 
    # ---------------------------------------------------------------
    if return_weights is True:
        return y_pred, w
    if return_weights is False:
        return y_pred



def model_3(X_test, X_train, y_train, deg, lam=1e-10, return_weights=False):
    """Trains linear regression model and returns predictions for test data.
        Model is given by: y = w_0 + w_1*x + w_2*x^2 + ... + w_D*x^D
    
    Args:
        X_test (numpy array): N x 1 array of test inputs
        X_train (numpy array): N x 1 array of training inputs
        y_train (numpy array): N x 1 array of training targets
        deg (int): Degree of polynomial
        lam (float, optional): Controls strength of regularization.. Defaults to 1e-10.
        return_weights (bool, optional): If True function also return model weights. Defaults to False.

    Returns:
        (np.array, np.array(optional)): Test set predictions and weights (if return_weights = True)
    """

    Phi_train = np.ones((X_train.shape[0],1))
    Phi_test = np.ones((X_test.shape[0],1))
    
    # ------------ add code here -----------------------------------
    for d in range(1,deg+1):
        Phi_train = np.concatenate((Phi_train, X_train**d), axis=1)
                                    #einsen wie shape Xtrain[0]
        Phi_test = np.concatenate((Phi_test, X_test**d), axis=1)
        

    Phi_test, Phi_train = normalize_features(Phi_test, Phi_train)
    
    w = lr.fit(Phi_train, y_train, lam)
    y_pred = lr.predict(Phi_test, w) 
    # ---------------------------------------------------------------

    if return_weights is True:
        return y_pred, w
    if return_weights is False:
        return y_pred