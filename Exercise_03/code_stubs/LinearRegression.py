import numpy as np

def fit(Phi, y, lam=1e-10):
    """Fits linear regression weights via normal equation.

    Args:
        Phi (numpy array):N x M design matrix
        y ([type]): N x 1 array of target values
        lam (float, optional): Controls strength of regularization. Defaults to 1e-10.

    Returns:
        numpy array: M x 1 array of weights
    """
    eye = np.eye(Phi.shape[1])
    eye[0,0] = 0
    #w = np.linalg.lstsq((Phi.T@Phi+lam*eye),Phi.T@y)
    #w = w[0]
    w = np.linalg.inv(Phi.T@Phi + lam*eye) @ Phi.T @ y
    return w

def predict(Phi,w):
    """Makes predition for linear regression model.

    Args:
        Phi (numpy array):N x M design matrix
        w (numpy array): M x 1 array of weights

    Returns:
        numpy array: N x 1 array of predictions
    """

    return Phi@w # <-- change this