import numpy as np
from sklearn import preprocessing

def normalize_features(X, mu, sigma):
    """Performs feature normalization s.t. columns in X have zero mean and unit variance.

    Args:
        X (numpy array): A NxD matrix with D features and N examples
        mu (numpy array): 1xD vector of mean values used for normalization
        sigma (numpy array): 1xD vector of standard deviations used for normalization

    Returns:
        numpy array: Normalized data.
    """

    #X_scaled = preprocessing.scale(X)
# ----------------- Add code here ----------------------
    ret = np.zeros(X.shape)
    for i in range(X.shape[1]):              #Anzahl Spalten
        ret[:,i] = (X[:,i] - mu[i])/sigma[i]


    
    #print(X)


    return ret # <-- change this
# -------------------------------------------------------

#return(X-mu)/sigma


def compute_rmse(y_true, y_pred):
    """Computes the root mean squared error (RMSE).

    Args:
        y_true (numpy array): vector of true values
        y_pred (numpy array): vector of predicted values

    Returns:
        float: root mean squred error.
    """
# ----------------- Add code here ----------------------
    rms = np.sqrt(((y_pred - y_true) ** 2).mean())

    return np.array(rms)
# -------------------------------------------------------
