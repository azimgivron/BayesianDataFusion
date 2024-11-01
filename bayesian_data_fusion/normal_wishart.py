import numpy as np
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal, wishart


class NormalWishart:
    def __init__(self, mu, kappa, T, nu):
        """Initializes the Normal-Wishart distribution.

        Parameters:
        mu : np.ndarray
            The mean vector.
        kappa : float
            The scaling factor.
        T : np.ndarray
            The scale matrix (precision matrix).
        nu : float
            The degrees of freedom.
        """
        self.dim = len(mu)
        self.zeromean = np.all(mu == 0)
        self.mu = mu
        self.kappa = kappa
        self.T = np.array(T)  # Ensure T is a numpy array
        self.nu = nu

    def rand(self):
        """Generates a random sample from the Normal-Wishart distribution.

        Returns:
        tuple : (mu_sample, Lambda_sample)
            A tuple containing:
            - mu_sample : np.ndarray
                A sample from the normal distribution.
            - Lambda_sample : np.ndarray
                A sample from the Wishart distribution.
        """
        Lam = wishart.rvs(df=self.nu, scale=self.T)  # Sample from Wishart distribution
        mu_sample = multivariate_normal.rvs(mean=self.mu, cov=np.linalg.inv(Lam) / self.kappa)
        return mu_sample, Lam
