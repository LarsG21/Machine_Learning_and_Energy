import numpy as np

def find_closest_center(X, centroids):
    """Find the closest center to each data point in X

    Args:
        X (np.array): N x D  input data
        centroids (np.array): k x D  centroids

    Returns:
        (np.array): N  index of closest centroid 
    """

    # initialize array of distance between X and centroids
    d = np.zeros((X.shape[0], centroids.shape[0]))
    #print(d.shape)
    # calculate distance from every point in X to every centroid
    # TODO: add code here 
    countX = 0
    countCentroids = 0
    for point in X:
        for centroid in centroids:
            d[countX][countCentroids] = np.linalg.norm(point-centroid)
            countCentroids += 1
        countCentroids = 0
        countX += 1

    a = np.argmin(d,axis=1)
    #print('X')
    #print(a)
    #print('X')
    # return the index with shortest distance
    return a # TODO: change code here

def update_centroids(X, z, k):
    """Return new estimate for centroids

    Args:
        X (np.array): N x D  input data
        z (np.array): N  index of closest centroid (from previous iteration) 
        k (integer): number of centroids

    Returns:
        (np.array): k x D  new estimate for centroids 
    """

    # initialize array of updated centroids
    centroids = np.zeros((k, X.shape[1]))

    for x in range(k):
        centroids[x,:] = np.mean(X[np.where(z == x)], axis=0)

    #print(centroids)
    # update centroids as the mean of the points assigned to them 
    # TODO: add code here 

    return centroids

def fit(X, init_centroids, max_iter=10):
    """Assign data points to clusters using k-Means

    Args:
        X (np.array): N x D  input data
        centroids (np.array): k x D  centroids 
        max_iter (integer): number of algorithm's maximum number of iterations 

    Returns:
        hist_centroids (list): list with each element being the (np.array) k x D centroid at each iteration
    """

    # initialize centroids (from previous iteration) 
    old_centroids = init_centroids

    # initialize history of centroids with initial value
    hist_centroids = [init_centroids]

    # calculate k from centroids' dimension
    k = init_centroids.shape[0]

    centroids = init_centroids

    for i in range(max_iter):
        print("Iteration {} ...".format(i+1))

        # TODO: add code here 
        #
        #
        z = find_closest_center(X,old_centroids)
        centroids = update_centroids(X,z,centroids.shape[0])



        # convergence criterion is if all elements of current centroids are equal to the centroids from previous iteration
        if (old_centroids==centroids).all():  # TODO: change code here
            print("k-Means converged in {} iterations".format(i+1))
            break
        
        # update old_centroid
        old_centroids = centroids

        # append newly calculate centroid to history
        hist_centroids.append(centroids)

    return hist_centroids

