import matplotlib.pyplot as plt
import numpy as np
import imageio
from scipy.stats import norm, multivariate_normal
from k_means import find_closest_center


def plot_joint(ax, mu, Sigma, lb, ub, point, samples=200):

    # prepare grid and calculate pdf values
    x_grid = np.linspace(lb,ub,samples)
    xx_1,xx_2 = np.meshgrid(x_grid, x_grid)
    zz_1 = norm.pdf(x_grid, loc=mu[0], scale=np.sqrt(Sigma[0,0]))
    zz_2 = norm.pdf(x_grid, loc=mu[1], scale=np.sqrt(Sigma[1,1]))
    zz_12 = multivariate_normal.pdf(np.dstack((xx_1, xx_2)), mean=mu, cov=Sigma)

    # plot into axis
    ax.plot(xs=x_grid, ys=np.ones_like(x_grid)*lb, zs=zz_1, label="Marginal density x1", color="tab:orange")
    ax.plot(xs=np.ones_like(x_grid)*lb, ys=x_grid, zs=zz_2, label="Marginal density x2", color="tab:green")
    ax.plot(xs=np.ones_like(x_grid)*lb, ys=np.ones_like(x_grid)*point, zs=np.linspace(0,np.max(zz_2)+1e-2,samples), label="Drawn sample x2={}".format(point), color="tab:red")
    ax.plot_wireframe(xx_1, xx_2, zz_12, alpha=0.8, label="Joint density", color="tab:blue")
    ax.view_init(35, 65)
    ax.set_xlim(lb,ub)
    ax.set_ylim(lb,ub)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    _ = ax.legend(loc="lower right")

def plot_conditional(ax, mu, Sigma, lb, ub, point, samples=200):

    # prepare grid and calculate pdf values for non-conditional distribution
    x_grid = np.linspace(lb,ub,samples)
    xx_1,_ = np.meshgrid(x_grid, x_grid)
    zz_1 = norm.pdf(x_grid, loc=mu[0], scale=np.sqrt(Sigma[0,0]))

    # calculate pdf values for conditional distribution
    zz_12 = multivariate_normal.pdf(np.dstack((xx_1, np.ones_like(xx_1)*point)), mean=mu, cov=Sigma)[0]
    zz_2 = norm.pdf(point, loc=mu[1], scale=np.sqrt(Sigma[1,1]))
    zz_1_cond = zz_12/zz_2
    
    # plot into axis
    ax.plot(x_grid, zz_1, label="Marginal density x1", color="tab:orange")
    ax.plot(x_grid, zz_1_cond, label="Marginal density x1 given x2={}".format(point), color="tab:purple")
    _ = ax.legend(loc="lower right")
        

def plot_dataset(ax, X, z=None, colors=None):

    if z is None:
        ax.scatter(X[:,0], X[:,1], marker='o', facecolors='none', color='gray')
    else:
        ax.scatter(X[:,0], X[:,1], edgecolors=[colors[i] for i in z], marker='o', facecolors='none')


def generate_gif(filenames, gifname):

    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(gifname, images, duration=1)

def plot_random_variables(ax, colors, cmaps, means, covs, x, y, marker_size=100, marker_width=5):
    
    # prepare grid
    pos = np.dstack((x, y))

    # put values in lists if only one variable
    if not isinstance(covs, list) and not isinstance(covs, tuple):
        means = means[np.newaxis,...]
        covs = [covs]
        colors = [colors]
        cmaps = [cmaps]

    # plot pdfs
    k = len(means)
    for i in range(k):
        rvs = multivariate_normal(means[i], covs[i])
        ax.scatter(means[i][0], means[i][1], marker='x', color=colors[i], s=marker_size, linewidths=marker_width)
        ax.contourf(x, y, rvs.pdf(pos), alpha=0.5, cmap=cmaps[i], levels=np.linspace(rvs.pdf(means[i][:] + np.sqrt(np.diag(covs[i]))), rvs.pdf(means[i][:]), num=5))