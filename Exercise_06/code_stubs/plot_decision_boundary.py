import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    """Plots 2D decision boundary of given model.

    Args:
        model (object): Model with a predict() method.
        X (numpy array): Input data
        y (numpy array): Targets
    """
    
    xx1,xx2 = np.meshgrid(np.linspace(np.min(X[:,0]),np.max(X[:,0]),50),np.linspace(np.min(X[:,1]),np.max(X[:,1]),50))
    XX = np.column_stack((np.ravel(xx1),np.ravel(xx2)))
    y_hat = np.reshape(model.predict(XX),(50,50))
    
    plt.figure(figsize=(9.5,8))
    plt.contour(xx1,xx2, y_hat, levels=[0.5], colors="black")
    c=plt.contourf(xx1, xx2, y_hat, cmap="RdBu", alpha=.4, levels=[0.0,0.2,0.4,0.6,.8,1.0])
    c=plt.contourf(xx1, xx2, y_hat, cmap="RdBu", alpha=.4, levels=np.arange(0,1.1,0.1))
    plt.colorbar(c)
    plt.scatter(*X[y==0].T, color="red", label="y=0")
    plt.scatter(*X[y==1].T, color="blue", label="y=1", marker="P")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.tight_layout()