import torch
import torch.distributions as dist
from typing import List, Union, Optional


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
    
    def expected_log(self, concentration: Optional[torch.Tensor] = None, 
                     rate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute E[log X] for the Gamma distribution.
        
        For X ~ Gamma(α, β):
        E[log X] = ψ(α) - log(β)
        
        where ψ is the digamma function (derivative of log gamma function).
        
        Args:
            concentration: Alternative concentration parameter (α) to compute expectation with.
                          If None, uses self.concentration.
            rate: Alternative rate parameter (β) to compute expectation with.
                  If None, uses self.rate.
        
        Returns:
            Expected value of log X
        """
        alpha = concentration if concentration is not None else self.concentration
        beta = rate if rate is not None else self.rate
        
        # E[log X] = ψ(α) - log(β)
        return torch.digamma(alpha) - torch.log(beta)

    def expected_log_prob(
        self, 
        concentration_p: torch.Tensor, 
        rate_p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute E_q[log p(X)] where q is self and p is Gamma(concentration_p, rate_p).
        
        This computes the expected log probability of X under distribution p,
        where the expectation is taken with respect to distribution q (self).
        
        For X ~ q = Gamma(α_q, β_q) and p = Gamma(α_p, β_p):
        
        log p(X) = α_p log β_p - log Γ(α_p) + (α_p - 1) log X - β_p X
        
        E_q[log p(X)] = α_p log β_p - log Γ(α_p) + (α_p - 1) E_q[log X] - β_p E_q[X]
        
        where:
        - E_q[log X] = ψ(α_q) - log(β_q)
        - E_q[X] = α_q / β_q
        
        Args:
            concentration_p: Concentration parameter α_p of distribution p
            rate_p: Rate parameter β_p of distribution p
            
        Returns:
            E_q[log p(X)]
        """
        alpha_p = concentration_p
        beta_p = rate_p
        
        # E_q[log X] using self (q) parameters
        e_q_log_x = self.expected_log()
        
        # E_q[X] using self (q) parameters
        e_q_x = self.mean()
        
        # E_q[log p(X)]
        result = (alpha_p * torch.log(beta_p) 
                 - torch.lgamma(alpha_p) 
                 + (alpha_p - 1) * e_q_log_x 
                 - beta_p * e_q_x)
        
        return result


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
    
    def expected_log(self, loc: Optional[torch.Tensor] = None,
                     covariance_matrix: Optional[torch.Tensor] = None,
                     scale_tril: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute E[log p(X)] for the Multivariate Gaussian distribution.
        
        Note: For a multivariate Gaussian, E[log X] doesn't have a simple closed form
        for the random variable X itself. This method computes the expected log-density
        E[log p(X)] which is related to the negative entropy.
        
        For X ~ N(μ, Σ):
        E[log p(X)] = -0.5 * [d * log(2π) + log|Σ| + d]
        
        where d is the dimensionality.
        
        Args:
            loc: Alternative mean vector. If None, uses self.loc.
            covariance_matrix: Alternative covariance matrix. If None, uses self.covariance_matrix.
            scale_tril: Alternative Cholesky factor. If None, uses self.scale_tril.
        
        Returns:
            Expected log-density value
        """
        # Use provided parameters or fall back to instance parameters
        use_loc = loc if loc is not None else self.loc
        use_cov = covariance_matrix if covariance_matrix is not None else self.covariance_matrix
        use_scale = scale_tril if scale_tril is not None else self.scale_tril
        
        # Create temporary distribution with specified parameters
        if use_scale is not None:
            temp_dist = dist.MultivariateNormal(use_loc, scale_tril=use_scale)
        else:
            temp_dist = dist.MultivariateNormal(use_loc, covariance_matrix=use_cov)
        
        # E[log p(X)] = -entropy
        return -temp_dist.entropy()

    def expected_log_prob(
        self,
        loc_p: torch.Tensor,
        covariance_matrix_p: Optional[torch.Tensor] = None,
        scale_tril_p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute E_q[log p(X)] where q is self and p is N(loc_p, Σ_p).
        
        This computes the expected log probability of X under distribution p,
        where the expectation is taken with respect to distribution q (self).
        
        For X ~ q = N(μ_q, Σ_q) and p = N(μ_p, Σ_p):
        
        log p(X) = -d/2 log(2π) - 1/2 log|Σ_p| - 1/2 (X - μ_p)ᵀ Σ_p⁻¹ (X - μ_p)
        
        E_q[log p(X)] = -d/2 log(2π) - 1/2 log|Σ_p| - 1/2 E_q[(X - μ_p)ᵀ Σ_p⁻¹ (X - μ_p)]
        
        Using the trace trick:
        E_q[(X - μ_p)ᵀ Σ_p⁻¹ (X - μ_p)] = tr(Σ_p⁻¹ Σ_q) + (μ_q - μ_p)ᵀ Σ_p⁻¹ (μ_q - μ_p)
        
        Args:
            loc_p: Mean vector μ_p of distribution p
            covariance_matrix_p: Covariance matrix Σ_p of distribution p
            scale_tril_p: Cholesky factor of Σ_p (alternative to covariance_matrix_p)
            
        Returns:
            E_q[log p(X)]
        """
        # Get q (self) parameters
        mu_q = self.loc
        d = mu_q.shape[0]  # Dimensionality
        
        # Get Σ_q (q's covariance)
        if self.scale_tril is not None:
            sigma_q = self.scale_tril @ self.scale_tril.T
        else:
            sigma_q = self.covariance_matrix
        
        # Get Σ_p (p's covariance)
        if scale_tril_p is not None:
            sigma_p = scale_tril_p @ scale_tril_p.T
        elif covariance_matrix_p is not None:
            sigma_p = covariance_matrix_p
        else:
            raise ValueError("Must provide either covariance_matrix_p or scale_tril_p")
        
        # Compute Σ_p⁻¹
        sigma_p_inv = torch.linalg.inv(sigma_p)
        
        # Compute log|Σ_p|
        sign, logdet_sigma_p = torch.linalg.slogdet(sigma_p)
        
        # Compute tr(Σ_p⁻¹ Σ_q)
        trace_term = torch.trace(sigma_p_inv @ sigma_q)
        
        # Compute (μ_q - μ_p)ᵀ Σ_p⁻¹ (μ_q - μ_p)
        diff = mu_q - loc_p
        mahalanobis_term = diff @ sigma_p_inv @ diff
        
        # E_q[log p(X)]
        result = (-0.5 * d * torch.log(torch.tensor(2 * torch.pi, dtype=mu_q.dtype, device=mu_q.device))
                 - 0.5 * logdet_sigma_p
                 - 0.5 * (trace_term + mahalanobis_term))
        
        return result


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
        samples = []
        for d in self.distributions:
            d_forward = d.forward()
            if isinstance(d_forward, list):
                # Handle nested ProductDistribution
                samples.extend([dist_obj.sample(sample_shape) for dist_obj in d_forward])  # type: ignore[attr-defined]
            else:
                samples.append(d_forward.sample(sample_shape))  # type: ignore
        return samples
    
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
        
        log_probs = []
        for d, v in zip(self.distributions, values):
            d_forward = d.forward()
            if isinstance(d_forward, list):
                # Handle nested ProductDistribution - not fully supported
                raise NotImplementedError("log_prob not implemented for nested ProductDistribution")
            else:
                log_probs.append(d_forward.log_prob(v))  # type: ignore
        return sum(log_probs)
    
    def expected_log(self, **kwargs) -> torch.Tensor:
        """
        Compute E[log X] for the product distribution.
        
        For independent distributions, E[log(X1 * X2 * ... * Xn)] = E[log X1] + E[log X2] + ... + E[log Xn]
        
        Args:
            **kwargs: Optional parameters to pass to each distribution's expected_log method.
                     Can include distribution-specific parameters.
        
        Returns:
            Sum of expected log values from all distributions
        """
        expected_logs = []
        for d in self.distributions:
            if hasattr(d, 'expected_log'):
                expected_logs.append(d.expected_log(**kwargs))
            else:
                raise NotImplementedError(f"Distribution {type(d)} does not implement expected_log")
        
        return sum(expected_logs)  # type: ignore[return-value]

    def expected_log_prob(self, distributions_p: List[Union['GammaDistribution', 'MultivariateGaussianDistribution', 'ProductDistribution']], **kwargs) -> torch.Tensor:
        """
        Compute E_q[log p(X)] where q is self (product of distributions) and p is another product.
        
        For independent distributions:
        E_q[log p(X₁, X₂, ...)] = E_q[log p(X₁)] + E_q[log p(X₂)] + ...
        
        Args:
            distributions_p: List of distributions for p, matching structure of self
            **kwargs: Additional parameters to pass to individual expected_log_prob methods
            
        Returns:
            Sum of expected log probabilities from all component distributions
        """
        if len(distributions_p) != len(self.distributions):
            raise ValueError(
                f"Number of distributions must match: "
                f"q has {len(self.distributions)}, p has {len(distributions_p)}"
            )
        
        expected_log_probs = []
        for q_dist, p_dist in zip(self.distributions, distributions_p):
            if hasattr(q_dist, 'expected_log_prob'):
                # Extract parameters from p_dist to pass to q_dist.expected_log_prob
                if isinstance(p_dist, GammaDistribution):
                    elp = q_dist.expected_log_prob(
                        concentration_p=p_dist.concentration,
                        rate_p=p_dist.rate
                    )
                elif isinstance(p_dist, MultivariateGaussianDistribution):
                    elp = q_dist.expected_log_prob(
                        loc_p=p_dist.loc,
                        covariance_matrix_p=p_dist.covariance_matrix,
                        scale_tril_p=p_dist.scale_tril
                    )
                elif isinstance(p_dist, ProductDistribution):
                    # Recursive call for nested products
                    elp = q_dist.expected_log_prob(distributions_p=p_dist.distributions)
                else:
                    raise NotImplementedError(f"expected_log_prob not implemented for {type(p_dist)}")
                
                expected_log_probs.append(elp)
            else:
                raise NotImplementedError(
                    f"Distribution {type(q_dist)} does not implement expected_log_prob"
                )
        
        return sum(expected_log_probs)  # type: ignore[return-value]
