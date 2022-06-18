import tensorflow as tf
import numpy as np

class NeuralNetwork:
    '''
        Represents a neural network with one hidden layer and binary classification

        Attributes:
            w1 (tensorflow.Variable): in_dim x n_neurons weight matrix between input and hidden layers
            w2 (tensorflow.Variable): n_neurons x out_dim weight matrix between hidden and output layers
            b1 (tensorflow.Variable): n_neurons x 1 bias between input and hidden layers
            b2 (tensorflow.Variable): out_dim x 1 bias between hidden and output layers
    ''' 
    def __init__(self, in_dim, out_dim, n_neurons, rng):

        # input parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_neurons = n_neurons

        # initialize parameters
        self.reset(rng)

    def reset(self, rng):

        # initialize weight matrices
        self.w1 = tf.Variable(rng.normal(0,5,size=(self.in_dim,self.n_neurons)), dtype=float) 
        self.w2 = tf.Variable(rng.normal(0,5,size=(self.n_neurons,self.out_dim)), dtype=float) 

        # initialize biases
        self.b1 = tf.Variable(np.zeros((1,self.n_neurons)), dtype=float) 
        self.b2 = tf.Variable(np.zeros((1,self.out_dim)), dtype=float) 

    
    def forward(self, x):
        """Calculate forward pass of NN

        Args:
            x (tensorflow.Variable): N x in_dim input data

        Returns:
            (tensorflow.Variable): N x out_dim result of forward pass
        """

        # load values from object's attributes
        w1 = self.w1
        w2 = self.w2
        b1 = self.b1
        b2 = self.b2

        # make static class method local
        sigmoid = NeuralNetwork.sigmoid

        # ----------- your code here
        z1 = sigmoid(x@w1+b1)
        z2 = sigmoid(z1@w2+b2)

        # --------------------------

        return z2  # <-- change this

    def loss(self, y_hat, y):
        """Calculate cross-entropy loss

        Args:
            y_hat (tensorflow.Variable): N x 1 predicted data
            y (tensorflow.Variable): N x 1 target data

        Returns:
            (tensorflow.Variable): cross-entropy loss
        """

        # get number of samples
        n = tf.shape(y_hat).numpy()[0]

        a = ((-y@tf.math.log(tf.transpose(y_hat))-(1-y)@tf.math.log(1-tf.transpose(y_hat))))   # <-- change this

        return a
        

    def update(self, loss, g, lr):
        """Update parameters of neural network

        Args:
            loss (tensorflow.Variable): cross-entropy loss 
            g (tensorflow.GradientTape): gradient tape containing gradients in relation to every parameter of NN
            lr (float): learning rate for gradient descent
        """

        # load values from object's attributes
        w1 = self.w1
        w2 = self.w2
        b1 = self.b1
        b2 = self.b2

        # gradients
        dLdw1 = g.gradient(loss, w1)
        dLdw2 = g.gradient(loss, w2)
        dLdb1 = g.gradient(loss, b1)
        dLdb2 = g.gradient(loss, b2)

        # gradient steps
        gw1 = dLdw1*lr                             #tf.Variable(np.zeros(tf.shape(w1).numpy()), dtype=float)  # <-- change this
        gw2 = dLdw2*lr  # <-- change this
        gb1 = dLdb1*lr  # <-- change this
        gb2 = dLdb2*lr  # <-- change this

        # perform gradient step for every parameter
        w1.assign_sub(gw1)              #w1 = w1 - gw2
        w2.assign_sub(gw2)
        b1.assign_sub(gb1)
        b2.assign_sub(gb2)
        
    def predict(self, x, t=.5):
        """Predict class given sample and threshold

        Args:
            x (tensorflow.Variable): N x in_dim input data
            t (float): decision threshold

        Returns:
            (float): N x 1 predicted class
        """

        # forward pass
        y_hat = self.forward(x).numpy() 

        return y_hat >= t  # <-- change this

    def fit(self, x, y, lr=1, n_epochs=10):
        """Train neural network's parameters

        Args:
            x (tensorflow.Variable): N x in_dim training data
            y (tensorflow.Variable): N x 1 target value
            lr (float): learning rate for gradient descent
            n_epochs(int): number of epochs for training
        """
        tf.random.set_seed(0)

        n_samples = tf.shape(x).numpy()[0]
        for epoch in range(n_epochs):
            for i in range(n_samples):
                with tf.GradientTape(persistent=True) as g:
                    loss = self.loss(self.forward(tf.expand_dims(x[i,:], 0)),tf.expand_dims(y[i], -1))

                self.update(loss, g, lr=lr)
                del g
            print(f"Epoch {epoch+1}/{n_epochs}")

    @staticmethod
    def sigmoid(z):

        return 1./(1 + tf.math.exp(-z))