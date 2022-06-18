import numpy as np
from scipy.stats import multivariate_normal

def init_parameters(X, k, seed=0):
    """Initialize model's parameters 

    Args:
        X (np.array): N x D  input data
        k (integer): number of gaussian distributions 

    Returns:
        means (np.array): k x D  mean of gaussian distributions
        covs (list): k-elements list with (np.array) D x D  covariance matrix of gaussian distributions
        pis (lsit): k-elements list with cluster probability of each cluster p(z_i = j)
    """

    # set random seed
    np.random.seed(seed)

    # initialize cluster probability equal to all clusters
    pis = [1/k]*k

    # initialize means as the value of a random point in the dataset
    means = np.array([X[np.random.choice(range(X.shape[0])), :] for i in range(k)])

    # initialize covariance matrices as the identity matrix
    covs = [np.eye(2) for i in range(k)]

    return means, covs, pis

def expectation_step(X, means, covs, pis):
    print('Test')
    """Expectation step

    Args:
        X (np.array): N x D  input data
        means (np.array): k x D  mean of gaussian distributions
        covs (list): k-elements list with (np.array) D x D  covariance matrix of gaussian distributions
        pis (lsit): k-elements list with cluster probability of each cluster p(z_i = j)

    Returns:
        gammas (np.array): N x k cluster assignment probabilities
    """
    print('Test')
    # calculate k from dimension of means
    k = len(means)
    #expectation step
    # TODO: add code here 
    z=[]
    for i in range(k):
        z.append(multivariate_normal.pdf(X, mean=means[i], cov=covs[i])*pis[i])
    z = np.array(z)
    n = np.sum(z, axis=0)
    gammas = np.divide(z,n).T
    
    return gammas  # TODO: change code here

def maximization_step(X, means, covs, pis, gammas):
    """Maximization step

    Args:
        X (np.array): N x D  input data
        means (np.array): k x D  mean of gaussian distributions
        covs (list): k-elements list with (np.array) D x D  covariance matrix of gaussian distributions
        pis (lsit): k-elements list with cluster probability of each cluster p(z_i = j)
        gammas (np.array): N x k cluster assignment probabilities
    """

    # calculate k from dimension of means
    k = len(means)

    # maximization step
    # TODO: add code here 
    for j in range(k):
            gamma = gammas[:, j]
            covs[j] = (gamma[np.newaxis, ...] * (X - means[j]).T) @ (X - means[j]) / np.sum(gamma)
            means[j] = np.sum(gamma[...,np.newaxis] * X, axis=0) / np.sum(gamma)
            pis[j] = np.mean(gamma)

    return  # Python passes objects as referece. It means that modifications on the arrays inside this function will reflect in the main notebook 

def fit(X, means, covs, pis, max_iter=50, eps=1e-6):
    """Optimizes Gaussian distributions' parameters for maximum likelihood

    Args:
        X (np.array): N x D  input data
        means (np.array): k x D  mean of gaussian distributions
        covs (list): k-elements list with (np.array) D x D  covariance matrix of gaussian distributions
        pis (lsit): k-elements list with cluster probability of each cluster p(z_i = j)
        max_iter (integer): number of algorithm's maximum number of iterations 
        eps (float): value used for convergence criterion

    Returns:
        hist_means (list): list with each element being the (np.array) k x D  means at each iteration
        hist_covs (list): list with each element being the k-elements list with (np.array) D x D  covariance matrices at each iteration
        hist_pis (list): list with each element being the k-elements list with cluster probabilities
    """

    # calculate k from dimension of means
    k = len(means)

    # initialize lists with iteration history 
    hist_means = [np.copy(means)]
    hist_covs = [[np.copy(cov) for cov in covs]]
    hist_pis = [np.array(pis)]

    for i in range(max_iter):
        print("Iteration {} ...".format(i+1))

        # save old values for convergence criterion
        old_means = np.copy(means)
        old_covs = np.copy(covs)
        old_pis = np.copy(pis)

        # perform em step
        # TODO: add code here 
        gammas = expectation_step(X, means, covs, pis)
        maximization_step(X, means, covs, pis, gammas)

        # convergence criterion is if maximum different of any element of means or covs is greater than eps
        if (np.max(np.max(np.abs(old_means-means))) and max([np.max(np.abs(old_covs[i]-covs[i])) for i in range(k)])) < eps:
            print("E-M converged in {} iterations".format(i+1))
            break

        # add values to history
        hist_means.append(np.copy(means))
        hist_covs.append([np.copy(cov) for cov in covs])
        hist_pis.append(np.copy(pis))

    return hist_means, hist_covs, hist_pis


