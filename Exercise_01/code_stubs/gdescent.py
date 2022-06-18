import numpy as np
from linesearch import linesearch 

def gdescent(fun,x0,N):
    '''Implementation of Gradient Descent algorithm

        Args:
            fun (Func): an object that provides values and gradients
            x0 (int,float): Starting point
            N (int): number of iterations

        Returns:
            x_N (numpy.array): 2x(N+1) array of visited points (including starting point)
            y_N (numpy.array): N+1 array of associated function values
    '''

    # allocate arrays
    x_N = np.zeros((2,N+1)) 
    y_N = np.zeros(N+1) 
    
    # initialize x_N,y_N on starting points
    x_N[:,0] = x0
    y_N[0] = fun.val(x0[0],x0[1])
    #y_N[0] = fun.val(*x0)

    x_k = x0  # initialize first point
    
    for i in range(1,N+1):
        # --------------------- YOUR CODE HERE ---------------------
        #if y_N[i-1] > y_N[i]: # nicht nötig wird schon in linesearch erledigt
        h = -fun.grad(x_k[0], x_k[1]) #h = -fun.grad(*x_k)  # obtain negative gradient from fun at point x_k
        t = linesearch(lambda t: fun.val(*(x_k+t*h)),0)
        x_k = x_k +t*h
        y_k = fun.val(x_k[0], x_k[1]) #y_k = fun.val(*x_k) # update current value for y_k
        # ----------------------------------------------------------

        # store visited points
        x_N[:,i] = x_k
        y_N[i] = y_k

    return x_N, y_N


    