import numpy as np

class Func:
    '''
    Represents a function

    Attributes:
        id (int): Specify which function to represent
    '''
    
    def __init__(self, id):
        if id in [0,1,2,3]:
            self.id = id
        else:
            raise ValueError('Unknown function.')
                       
    def val(self,x1,x2):
        '''Evaluates function
    
            Args:
                x1 (float, numpy.array): first argument of function
                x2 (float, numpy.array): second argument of function
    
            Returns:
                f (float, numpy.array): evaluation of the function at (x1,x2)
        '''
        if self.id == 0:
            return np.exp(np.power((x1-2),2)/5+(np.power((x2-3),2)/10 -1)) # <-- replace with code for value of f_1
        elif self.id == 1:
            return np.sin(np.pi*(x1-1))+np.sin(np.pi*(x2-1)) # <-- replace with code for value of f_2
                   
        elif self.id == 2:
            return Func.__h(x1,x2,2,1,2,np.pi/8)
        elif self.id == 3:
            return Func.__h(x1,x2,50,3,2,-np.pi/8)
        
        
    def grad(self,x1,x2):
        '''Evaluates gradient
    
            Args:
                x1 (float, numpy.array): first argument of function
                x2 (float, numpy.array): second argument of function
    
            Returns:
                g (float, numpy.array): evaluation of the gradient of the function at (x1,x2)
        '''
        
        if not np.isscalar(x1) or not np.isscalar(x2):
            raise ValueError('x1 or x2 are not scalars!')
        else:
            if self.id == 0:
                return np.array([(2*(x1-2)/5)*np.exp(np.power((x1-2),2)/5+(np.power((x2-3),2)/10 -1)),((x2-3)/5)*np.exp(np.power((x1-2),2)/5+(np.power((x2-3),2)/10 -1))]) # <-- replace with code for gradient of f_1
            elif self.id == 1:
                return np.array([np.pi*np.cos(np.pi*(x1-1)),np.pi*np.cos(np.pi*(x2-1))]) # <-- replace with code for gradient of f_2
            elif self.id == 2:
                return Func.__g_h(x1,x2,2,1,2,np.pi/8)
            elif self.id == 3:
                return Func.__g_h(x1,x2,50,3,2,-np.pi/8)


    @staticmethod
    def __h(x1,x2,lam,c1,c2,a):
        return (np.cos(a)**2 + np.sin(a)**2 * lam)*(x1-c1)**2 + (np.cos(a)**2 * lam + np.sin(a)**2)*(x2-c2)**2 + \
        2*np.cos(a)*np.sin(a)*(1-lam)*(x1-c1)*(x2-c2)
    
    @staticmethod
    def __g_h(x1,x2,lam,c1,c2,a):
        return np.array([(np.cos(a)**2 + np.sin(a)**2 * lam)*2*(x1-c1) + 2*np.cos(a)*np.sin(a)*(1-lam)*(x2-c2),\
        (np.cos(a)**2 * lam + np.sin(a)**2)*2*(x2-c2) + 2*np.cos(a)*np.sin(a)*(1-lam)*(x1-c1)])
