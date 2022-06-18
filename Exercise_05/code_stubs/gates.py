import numpy as np


class MultiplyGate:
    '''
        Represents a multiplication gate of a computation graph

        Attributes:
            x (float): First operand of forward pass
            y (float): Second operand of forward pass
    ''' 

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        """Calculate forward pass of multiplication gate with 2 inputs

        Args:
            x (float): first operand
            y (float): second operand

        Returns:
            (float): result of forward pass
        """
        
        # store values into object's attributes
        self.x = x
        self.y = y

        return x*y  # <-- change this
    
    def backward(self,dz):
        """Calculate backward pass of multiplication gate with 2 inputs

        Args:
            dz (float): partial derivative of computation graph's function in relation to gate's output 

        Returns:
            (list): 2-elements list with the partial derivative of computation graph's function in relation to both operands
        """

        # load values from object's attributes
        x = self.x
        y = self.y

        # calculate partial derivatives
        dx = y*dz  # <-- change this
        dy = x*dz  # <-- change this

        return [dx, dy]


class SumGate:
    '''
        Represents a sum gate of a computation graph

        Attributes:
            x (float): First operand of forward pass
            y (float): Second operand of forward pass
    '''

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        """Calculate forward pass of sum gate with 2 inputs

        Args:
            x (float): first operand
            y (float): second operand

        Returns:
            (float): result of forward pass
        """

        # store values into object's attributes
        self.x = x
        self.y = y

        return x+y  # <-- change this
    
    def backward(self,dz):
        """Calculate backward pass of sum gate with 2 inputs

        Args:
            dz (float): partial derivative of computation graph's function in relation to gate's output 

        Returns:
            (list): 2-elements list with the partial derivative of computation graph's function in relation to both operands
        """
        
        # load values from object's attributes
        x = self.x
        y = self.y

        # calculate partial derivatives
        dx = 1*dz  # <-- change this
        dy = 1*dz  # <-- change this

        return [dx, dy]


class SigmoidGate:
    '''
        Represents a sigmoid gate of a computation graph

        Attributes:
            x (float): Operand of forward pass
    '''

    def __init__(self):
        self.x = None

    def forward(self,x):
        """Calculate forward pass of sigmoid gate

        Args:
            x (float): operand

        Returns:
            (float): result of forward pass
        """

        # store values into object's attributes
        self.x = x

        # make static class method local
        sigmoid = SigmoidGate.sigmoid

        return 1/(1+np.exp(-x))  # <-- change this
    
    def backward(self,dz):
        """Calculate backward pass of sigmoid gate

        Args:
            dz (float): partial derivative of computation graph's function in relation to gate's output 

        Returns:
            (list): partial derivative of computation graph's function in relation to operand
        """

        # load values from object's attributes
        x = self.x

        # make static class method local
        sigmoid = SigmoidGate.sigmoid

        # calculate partial derivatives
        dx = 0  # <-- change this

        return SigmoidGate.forward(self,x)*(1-SigmoidGate.forward(self,x))*dz

    @staticmethod
    def sigmoid(z):
        return 1./(1+np.exp(-z))