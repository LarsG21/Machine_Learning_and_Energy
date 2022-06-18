import numpy as np


def transform(X, n_components):
    """Applies PCA
   
    Args:
        X (N x D array): original data
        n_components (int): number of principal components

    Returns:
        W (N x n_components array): transformed data
        U (D x D array): matrix of eigenvectors
        eig_vals (array of length D): eigenvalues
    """
    # -------- add code here -----------------
    mu = np.mean(X)
    sigma = np.std(X)
    cov_mtx = np.cov(X,rowvar=False)
    print(X.shape)
    print(cov_mtx.shape)

    eigenvalues, eig_valseig_vect_mtx = np.linalg.eig(cov_mtx)

    transformed = mu
    for i in range(eig_valseig_vect_mtx.shape[0]):
        transformed += eig_valseig_vect_mtx[i]*eig_valseig_vect_mtx[i].T*(X-mu)

   
    U = eig_valseig_vect_mtx # <-- change this
    W = X@U[:,0:n_components]
    eig_vals = eigenvalues
    # ----------------------------------------
    return W, U, np.sort(eig_vals)[::-1]


def backtransform(W, U):
    """Applies inverse transformation.

    Args:
        W (N x n_components array): transformed data
        U (D x D array): matrix of eigenvectors

    Returns:
        X_rec (N x D array): recovered data
    """
    X_rec = np.ones((len(W),len(U))) # <-- change this
    return X_rec
