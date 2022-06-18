import numpy as np

def early_stopping(model, X_train, y_train, X_val, y_val, n_epochs, batch_size, patience):
    """Trains Keras model using early stopping.

    Args:
        model (Keras model object): A compiled Keras model.
        X_train (numpy array): Inputs for training
        y_train (numpy array): Target values for training
        X_val (numpy array): Inputs for validation
        y_val (numpy array): Target values for validation
        n_epochs (int): Maximum number of training epochs
        batch_size (int): Batch size.
        patience (int): Patience parameter. Defines how long to continue training without improvement.

    Returns:
        train_loss, val_loss (list, list): Training and validation losses.
    """
    train_loss = []
    val_loss = []

    # --------- add code here --------------
    loss_increase_counter = 0
    
    model.fit(X_train,y_train,epochs=1, verbose=0, batch_size=batch_size)
    train_loss.append(model.test_on_batch(X_train,y_train))
    absmin = model.test_on_batch(X_val,y_val)
    val_loss.append(absmin)
    for i in range(1,n_epochs):
        if(loss_increase_counter >= patience):
            break
            
        model.fit(X_train,y_train,epochs=1, verbose=0, batch_size=batch_size)
        train_loss.append(model.test_on_batch(X_train,y_train))
        val_loss.append(model.test_on_batch(X_val,y_val))

        if(val_loss[i] < absmin):
            absmin = val_loss[i]
            weights = model.get_weights()
            loss_increase_counter = 0
        else:
            loss_increase_counter += 1

        
    
    model.set_weights(weights)

    # --------------------------------------
    return train_loss, val_loss
