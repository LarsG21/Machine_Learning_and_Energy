import numpy as np
from sklearn.linear_model import LinearRegression

def lin_reg_benchmark(data_train, data_test):
    """Linear regression benchmark model.

    Args:
        data_train (pandas df): training data
        data_test (pandas df): test data

    Returns:
        (numpy array): predictions
    """

    # prepare data
    X_train = data_train.loc[:,["windspeed100"]].values
    X_test = data_test.loc[:,["windspeed100"]].values
    y_train = data_train["power_generation"].values
    
    # fit model
    mdl = LinearRegression()
    mdl.fit(X_train, y_train)
    
    # predict test set
    y_hat = mdl.predict(X_test)
    return np.clip(y_hat, 0.0, 1.0)
