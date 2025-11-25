import torch
import torch.distributions as dist
from typing import List, Union


class GammaDistribution:
    """
    Gamma distribution wrapper that provides forward sampling and entropy computation.
    
    The forward method returns the torch Gamma distribution.
    The entropy method computes the differential entropy of the Gamma distribution.
    """
    
    def __init__(self, concentration, rate):
        """
        Initialize Gamma distribution.
        
        Args:
            concentration: Shape parameter (alpha) of the Gamma distribution
            rate: Rate parameter (beta) of the Gamma distribution
        """
        self.concentration = concentration
        self.rate = rate
        self._distribution = None
    
    def update_parameters(self, concentration=None, rate=None):
        """
        Update distribution parameters and invalidate cached distribution.
        
        Args:
            concentration: New concentration parameter (optional)
            rate: New rate parameter (optional)
        """
        if concentration is not None:
            self.concentration = concentration
        if rate is not None:
            self.rate = rate
        self._distribution = None
    
    def forward(self):
        """
        Return the torch Gamma distribution.
        
        Returns:
            torch.distributions.Gamma instance
        """
        if self._distribution is None:
            self._distribution = dist.Gamma(self.concentration, self.rate)
        return self._distribution
    
    def mean(self):
        """
        Compute the mean (expected value) of the Gamma distribution.
        
        For Gamma(concentration, rate): E[X] = concentration / rate
        
        Returns:
            Mean value (scalar or tensor)
        """
        return self.concentration / self.rate
    
    def entropy(self):
        """
        Compute the differential entropy of the Gamma distribution.
        
        The entropy H = -E[log p(X)] measures the uncertainty in the distribution.
        
        Returns:
            Entropy value (scalar or tensor)
        """
        return self.forward().entropy()


class MultivariateGaussianDistribution:
    """
    Multivariate Gaussian (Normal) distribution wrapper.
    
    Provides forward sampling and entropy computation.
    """
    
    def __init__(self, loc, covariance_matrix=None, scale_tril=None):
        """
        Initialize Multivariate Gaussian distribution.
        
        Args:
            loc: Mean vector
            covariance_matrix: Covariance matrix (optional, mutually exclusive with scale_tril)
            scale_tril: Lower triangular Cholesky factor of covariance (optional)
        """
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.scale_tril = scale_tril
        self._distribution = None
    
    def update_parameters(self, loc=None, covariance_matrix=None, scale_tril=None):
        """
        Update distribution parameters and invalidate cached distribution.
        
        Args:
            loc: New mean vector (optional)
            covariance_matrix: New covariance matrix (optional)
            scale_tril: New Cholesky factor (optional)
        """
        if loc is not None:
            self.loc = loc
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        if scale_tril is not None:
            self.scale_tril = scale_tril
        self._distribution = None
    
    def forward(self):
        """
        Return the torch MultivariateNormal distribution.
        
        Returns:
            torch.distributions.MultivariateNormal instance
        """
        if self._distribution is None:
            if self.scale_tril is not None:
                self._distribution = dist.MultivariateNormal(
                    self.loc, scale_tril=self.scale_tril
                )
            else:
                self._distribution = dist.MultivariateNormal(
                    self.loc, covariance_matrix=self.covariance_matrix
                )
        return self._distribution
    
    def mean(self):
        """
        Compute the mean (expected value) of the Multivariate Gaussian.
        
        Returns:
            Mean vector
        """
        return self.loc
    
    def entropy(self):
        """
        Compute the differential entropy of the Multivariate Gaussian.
        
        Returns:
            Entropy value (scalar or tensor)
        """
        return self.forward().entropy()


class ProductDistribution:
    """
    Product of independent probability distributions.
    
    The forward method computes the product of distributions.
    The entropy method computes the sum of individual entropies (since they're independent).
    Supports nesting of ProductDistributions.
    """
    
    def __init__(self, distributions: List[Union[GammaDistribution, MultivariateGaussianDistribution, 'ProductDistribution']]):
        """
        Initialize product distribution from a list of distributions.
        
        Args:
            distributions: List of distribution objects (can include other ProductDistributions)
        """
        self.distributions = distributions
    
    def forward(self):
        """
        Return the product of distributions as an Independent distribution.
        
        For truly independent distributions, we stack them and use Independent wrapper.
        
        Returns:
            List of torch distribution objects
        """
        return [d.forward() for d in self.distributions]
    
    def mean(self):
        """
        Compute the mean for each distribution in the product.
        
        Returns:
            List of mean values, one for each distribution
        """
        return [d.mean() for d in self.distributions]
    
    def entropy(self):
        """
        Compute the total entropy as sum of individual entropies.
        
        For independent distributions: H(X1, X2, ..., Xn) = H(X1) + H(X2) + ... + H(Xn)
        
        Returns:
            Total entropy (scalar or tensor)
        """
        entropies = [d.entropy() for d in self.distributions]
        return sum(entropies)
    
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from all distributions in the product.
        
        Args:
            sample_shape: Shape of samples to generate
            
        Returns:
            List of samples from each distribution
        """
        return [d.forward().sample(sample_shape) for d in self.distributions]
    
    def log_prob(self, values: List):
        """
        Compute log probability for a list of values.
        
        Args:
            values: List of values, one for each distribution
            
        Returns:
            Sum of log probabilities
        """
        if len(values) != len(self.distributions):
            raise ValueError("Number of values must match number of distributions")
        
        log_probs = [d.forward().log_prob(v) for d, v in zip(self.distributions, values)]
        return sum(log_probs)
