import numpy as np
from scipy.stats import norm
from scipy import special

def pdf(x, mu=0.0, sigma=1.0):
    """Calculate PDF of a univariate normal distribution.

    Args:
        x (float or array): Points to evalute the function.
        mu (float, optional): mean. Defaults to 0.0.
        sigma (float, optional): standard devation. Defaults to 1.0.

    Returns:
        float or array: Values of PDF at x.
    """
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (x - mu))**2))

    return y # <-- change this



def cdf(x, mu=0.0, sigma=1.0):
    """Calculate CDF of a univariate normal distribution.

    Args:
        x (float or array): Points to evalute the function.
        mu (float, optional): mean. Defaults to 0.0.
        sigma (float, optional): standard devation. Defaults to 1.0.

    Returns:
        float or array: Values of CDF at x.
    """
    return  0.5*(1+special.erf((x-mu)/(sigma*np.sqrt(2)))) # <-- change this



def sample(mu=0.0, sigma=1.0, n_samples=1, random_state=None):
    """Sample from univaraite normal distribution.

    Args:
        mu (float, optional): Mean. Defaults to 0.0.
        sigma (float, optional): Standard deviatioon. Defaults to 1.0.
        n_samples (int, optional): Number of samples to be drawn. Defaults to 1.
        random_state (int, optional): Random seed. Defaults to None.

    Returns:
        array: random samples from distribution.
    """
    return norm.rvs(size=n_samples,loc=mu,scale=sigma, random_state=random_state)



def estimate_mu(x):
    """Estimate mean from samples.

    Args:
        x (array): Samples

    Returns:
        float: Estimated mean.
    """
    sum = 0
    for i in x:
        sum += i

    sum = sum/x.size

    return sum # <-- change this



def estimate_sigma(x):
    """Estimate standard deviation from samples.

    Args:
        x (array): Samples.

    Returns:
        float: Estimated standard deviation.
    """
    mean = estimate_mu(x)
    sum = 0
    for i in x:
       sum  += (mean-i)**2

    #sum = (mean-x)**2
    sum = sum/x.size
    sum = np.sqrt(sum)

    return sum # <-- change this