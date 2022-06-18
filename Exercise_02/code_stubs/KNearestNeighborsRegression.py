import numpy as np
import my_utils


def get_neighbors_idx(v, M, n_neighbors):
    """Get indeces of nearest neighbors according to L2 norm.

    Args:
        v (numpy array): A 1xD vector of real values
        M (numpy array): A NxD matrix of real values
        n_neighbors (int): Number of neighbors

    Returns:
        numpy array: Row indices of nearest neighbors
    """
    dimentions = M.shape
    number_of_vectors = dimentions[0]
    distances = []

    for e in range(number_of_vectors):
        x = np.linalg.norm((M[e]-v))
        distances.append(x)
    newOrder = np.argsort(distances,kind='quicksort')

    finalOrder = []
    
    for e in range(n_neighbors):
        finalOrder.append(newOrder[e])


    #print(newOrder)
    #print(finalOrder)

    return np.array(finalOrder)


def knn_regressor(X_test, X_train, y_train, n_neighbors):
    """Perform KNN regression on samples in X_test.

    Args:
        X_test (numpy array): M x D matrix of test points
        X_train (numpy array): N x D matrix of test points
        y_train (numpy array): N x 1 Array of target values
        n_neighbors (int): Number of neighbors

    Returns:
        y_pred (numpy array): M x 1 vector of predicted values
    """
    N_EXAMPLES = X_test.shape[0]
    y_pred = np.zeros((N_EXAMPLES,1))           #Erste dimension immer die Examples und dann die features
# ----------------- Add code here ----------------------
    for i in range(N_EXAMPLES):
        indexes = get_neighbors_idx(X_test[i],X_train,n_neighbors)
        a = y_train[indexes]
        b = np.average(a)   #mean of the n nearest neighbours
        y_pred[i,0] = b
        #y_pred[i,0] = np.means(y_train[idx])

    return y_pred
# -------------------------------------------------------
   



def knn_crossval(X, y, neighbors, n_folds, seed=42):
    """Performs cross validation for knn_regressor.

    Args:
        X (numpy array): N x D matrix of inputs
        y (numpy array): N x 1 Array of target values
        neighbors (list): Possible values for number of neighbors
        n_folds (int): Number of folds

    Returns:
        numpy array: Array of average RMSE values corresponding to values given in neighbors
    """
    # HINT: you can create an array of booleans (i.e. True and False) by comparing
    # a scalar s and a vector v using "s==v"" and "s!=v".
    # You can then use the resulting boolean array to slice arrays.
    # This following code create a vector of length N with an equal amount of integers from the set {0,1,2,..., n_folds-1}:
    rs = np.random.RandomState(seed=seed) # fix random seed
    idx = rs.permutation(np.repeat(np.arange(0,n_folds), np.int(np.ceil(len(X)/n_folds)))[0:len(X)])


    rmse_cv = np.zeros(len(neighbors))
# ----------------- Add code here ----------------------
    for i in range(len(neighbors)):
        rsme_fold = np.zeros(n_folds)
        for j in range(n_folds):
            y_pred = knn_regressor(X_test= X[j==idx,:],
                                   X_train= X[j!=idx,:],          
                                   y_train= y[j!=idx,:],
                                   n_neighbors= neighbors[i])
            rsme_fold[j] = my_utils.compute_rmse(y[j==idx,:],y_pred)
        rmse_cv[i] = np.mean(rsme_fold)
# -------------------------------------------------------
    return rmse_cv