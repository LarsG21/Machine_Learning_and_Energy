def linesearch(fun,x0):
    '''Search for minimum of 1D function via interval doubling

        Args:
            fun (function): a function of the form f(x)
            x0 (int,float): Starting point

        Returns:
            t: Step taken until (approximate) minimum
    '''
    
    
    maxiter = 25  # max number of iterations
    t = 1e-4 # initial step size
    val = fun(x0 + t)  # current function value
    iter = 1 # iteration counter

    while True:
        x0+=t
        t=t*2
        newval =  fun(x0+t)
        if val < newval or iter >= maxiter:
            break  
        val = newval 
        iter += 1
        

        # ----------------------------------------------------------         
    
    return t