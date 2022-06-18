import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def build_linreg_model(dim_in=1, learning_rate=0.1):
    """Builds Keras linear regression model.

    Args:
        dim_in (int, optional): Dimensionality of input. Defaults to 1.
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.1.

    Returns:
        mdl (Keras object): Compiled model.
    """
    # define model
    x = layers.Input(shape=(dim_in,), name="InputLayer")
    y = layers.Dense(units=1, name="OutputLayer")(x)
    mdl = keras.Model(inputs=x, outputs=y, name="linreg_model")
    
    # compile model
    mdl.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss="mse")
    return mdl



def build_logreg_model(dim_in=2, learning_rate=5.0):
    """Builds Keras logistic regression model.

    Args:
        dim_in (int, optional): Dimensionality of input. Defaults to 2.
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 5.0.

    Returns:
        mdl (Keras object): Compiled model.
    """
    #define model
    # --------- add code here --------------
    x = layers.Input(shape=(dim_in,), name="InputLayer")
    y = layers.Dense(units=1, name="OutputLayer")(x)                    #w_o +x1*w_1 + x2*w_2
    sigmoid = layers.Activation('sigmoid', name='layers_sigmoid_function')(y)   # y = sigmoid(z)   bzw sigmoid = sigmoid(y) für unseren fall
    mdl = keras.Model(inputs=x, outputs=sigmoid, name="logreg_model")


    # --------------------------------------

    mdl.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),loss="binary_crossentropy") # <-- uncomment this
    return mdl




def build_NN_model(dim_in, n_neurons, optimizer="adam"):
    """Builds Keras neural network model.

    Args:
        dim_in (int): Dimensionality of input.
        n_neurons (int): Number of neurons in hidden layers.
        optimizer (str or Keras object, optional): Optimizer. Defaults to "adam".

    Returns:
        mdl (Keras object): Compiled model.
    """
    # --------- add code here --------------
    #Input
    L1 = layers.Input(shape=(dim_in,), name="InputLayer")
    #Hidden Layers
    L2 = layers.Dense(units=n_neurons, activation ='relu', name="y")(L1)
    L3 = layers.Dense(units=n_neurons, activation ='relu', name="z")(L2)
    L4 = layers.Dense(units=n_neurons, activation ='relu', name="a")(L3)
    #Output Layer
    L5 = layers.Dense(units=1, activation ='sigmoid', name="b")(L4)

    #Create Model
    mdl = keras.Model(inputs=L1, outputs=L5, name="logreg_model" )

    #Compile Model
    mdl.compile(optimizer= optimizer ,loss="binary_crossentropy") # <-- uncomment this
  
    # --------------------------------------
    
    return mdl
