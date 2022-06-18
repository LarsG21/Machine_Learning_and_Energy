import numpy as np

def normalize_features(X, mu, sigma):
    """Performs feature normalization s.t. columns in X have zero mean and unit variance.

    Args:
        X (numpy array): A NxD matrix with D features and N examples
        mu (numpy array): 1xD vector of mean values used for normalization
        sigma (numpy array): 1xD vector of standard deviations used for normalization

    Returns:
        numpy array: Normalized data.
    """
    return (X-mu)/sigma


def compute_rmse(y_true, y_pred):
    """Computes the root mean squared error (RMSE).

    Args:
        y_true (numpy array): vector of true values
        y_pred (numpy array): vector of predicted values

    Returns:
        float: root mean squred error.
    """
    return np.sqrt(np.mean(np.square(y_true-y_pred)))