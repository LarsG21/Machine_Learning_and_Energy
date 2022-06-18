import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def build_model(dim_in=(64,64,3), n_classes=11):
    """Builds a CNN model.

    Model description:

    - Input: 64x64x3 image
                |
                V
    - Convolutional Layer: 16 filters, 8x8 kernel, relu activation
    - Max Pooling: 4x4
                |
                V
    - Convolutional Layer: 32 filters, 4x4 kernel, relu activation
    - Max Pooling: 2x2
                |
                V
    - Convolutional Layer: 64 filters, 4x4 kernel, relu activation
    - Max Pooling: 2x2
                |
                V    
    - Flatten Layer
    - Dense Layer: 11 neurons (this is the number of classes), softmax activation <-- outout layer

    Args:
        dim_in (tuple, optional): Dimensionality of the input image. Defaults to (64,64,3).
        n_classes (int, optional): Number of classes in the data. Defaults to 11.

    Returns:
        model (Keras model object): Compiled Keras model.
    """
    # --------- add code here --------------
    x = layers.Input(dim_in)
    L1 = layers.Conv2D(filter = 16, (8,8), activation='relu')(x)
    L2 = layers.MaxPool2D((4,4))(L1)
    L3 = layers.Conv2D(32, (4,4), activation='relu')(L2)
    L4 = layers.MaxPool2D((2,2))(L3)
    L5 = layers.Conv2D(64, (4,4), activation='relu')(L4)
    L6 = layers.MaxPool2D((2,2))(L5)
    L7 = layers.Flatten()(L6)
    L8 = layers.Dense(units = 11, activation ='softmax')(L7)
    model = keras.Model(inputs=x, outputs=L8, name="NEW_MODEL")

    # --------------------------------------
    
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics="accuracy") # <-- uncomment this

    return model