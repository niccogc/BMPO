# type: ignore
import jax as jnp
import torch
import torch.distributions as dist
import quimb.tensor as qt
import numpy as np
from typing import List, Union, Optional, Any
torch.set_default_dtype(torch.float32)   # or torch.float64


def _extract_data(param: Any, dtype=torch.float64) -> torch.Tensor:
    """Helper to extract torch tensor data from potential quimb.Tensor."""
    if isinstance(param, qt.Tensor):
        # We assume the data inside quimb tensor might already be torch or numpy
        data = param.data
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=dtype)
        elif isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)
    elif isinstance(param, np.ndarray):
        return torch.tensor(param, dtype=dtype)
    elif isinstance(param, torch.Tensor):
        return param.to(dtype=dtype)
    else:
        return torch.tensor(param, dtype=dtype)

def _wrap_if_quimb(data: torch.Tensor, reference: Any, backend: str) -> Any:
    """
    Wraps the result in a quimb.Tensor if the reference was a quimb.Tensor.
    Preserves indices.
    """
    data = to_backend(data, backend)
    if isinstance(reference, qt.Tensor):
        # We store the torch tensor (potentially with grads) directly in the quimb tensor data
        return qt.Tensor(data=data, inds=reference.inds, tags=reference.tags)
    return data

def to_backend(data, backend: str):
    if backend == 'torch':
        return data

    # Convert to numpy first (handles CPU/GPU detachment)
    arr = data.detach().cpu().numpy() if hasattr(data, 'detach') else data

    if backend == 'numpy':
        return arr
    elif backend == 'jax':
        import jax.numpy as jnp
        return jnp.array(arr)
    
    raise ValueError(f"Unsupported backend: {backend}")

class GammaDistribution:
    """
    Gamma distribution wrapper that provides forward sampling and entropy computation.
    Supports quimb.Tensor parameters and returns expectations as quimb.Tensors.
    """
    
    def __init__(self, concentration, rate, backend = 'torch'):
        """
        Initialize Gamma distribution.
        
        Args:
            concentration: Shape parameter (alpha). Can be quimb.Tensor.
            rate: Rate parameter (beta). Can be quimb.Tensor.
        """
        self.backend = backend
        self.concentration = concentration
        self.rate = rate
        self._distribution = None
    
    def update_parameters(self, concentration=None, rate=None):
        if concentration is not None:
            self.concentration = concentration
        if rate is not None:
            self.rate = rate
        self._distribution = None
    
    def forward(self):
        if self._distribution is None:
            c_data = _extract_data(self.concentration)
            r_data = _extract_data(self.rate)
            self._distribution = dist.Gamma(c_data, r_data)
        return self._distribution
    
    def mean(self):
        """
        Returns E[X] = alpha / beta.
        If inputs are quimb.Tensors, returns a quimb.Tensor with the same bond indices.
        """
        c_data = _extract_data(self.concentration)
        r_data = _extract_data(self.rate)
        res = c_data / r_data
        return _wrap_if_quimb(res, self.concentration, self.backend)
    
    def entropy(self):
        """Returns entropy. If inputs are quimb.Tensors, returns quimb.Tensor."""
        res = self.forward().entropy()
        return _wrap_if_quimb(res, self.concentration, self.backend)
    
    def expected_log(self, concentration: Optional[Any] = None, 
                     rate: Optional[Any] = None) -> Union[torch.Tensor, qt.Tensor]:
        """Returns E[log X] wrapped in quimb.Tensor if applicable."""
        conc_obj = concentration if concentration is not None else self.concentration
        rate_obj = rate if rate is not None else self.rate
        
        alpha = _extract_data(conc_obj)
        beta = _extract_data(rate_obj)
        
        res = torch.digamma(alpha) - torch.log(beta)
        
        # Use conc_obj as reference for indices
        ref = conc_obj if isinstance(conc_obj, qt.Tensor) else self.concentration
        return _wrap_if_quimb(res, ref, self.backend)

    def expected_log_prob(self, concentration_p: Any, rate_p: Any) -> Union[torch.Tensor, qt.Tensor]:
        """Returns E_q[log p(X)] wrapped in quimb.Tensor if applicable."""
        alpha_p = _extract_data(concentration_p)
        beta_p = _extract_data(rate_p)
        
        # We need the raw data for internal calculation to avoid double wrapping
        # Call expected_log and mean but handle unwrapping if they return qt.Tensor
        e_q_log_x_obj = self.expected_log()
        e_q_x_obj = self.mean()
        
        e_q_log_x = e_q_log_x_obj.data if isinstance(e_q_log_x_obj, qt.Tensor) else e_q_log_x_obj
        e_q_x = e_q_x_obj.data if isinstance(e_q_x_obj, qt.Tensor) else e_q_x_obj
        
        result = (alpha_p * torch.log(beta_p) 
                 - torch.lgamma(alpha_p) 
                 + (alpha_p - 1) * e_q_log_x 
                 - beta_p * e_q_x)
        
        return _wrap_if_quimb(result, self.concentration, self.backend)


class MultivariateGaussianDistribution:
    """
    Multivariate Gaussian (Normal) distribution wrapper.
    Supports quimb.Tensor parameters.
    """
    
    def __init__(self, loc, covariance_matrix=None, scale_tril=None, backend='torch'):
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.scale_tril = scale_tril
        self._distribution = None
    
    def update_parameters(self, loc=None, covariance_matrix=None, scale_tril=None):
        if loc is not None:
            self.loc = loc
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        if scale_tril is not None:
            self.scale_tril = scale_tril
        self._distribution = None
    
    def _get_params(self):
        loc_data = _extract_data(self.loc)
        if loc_data.ndim > 1:
            loc_data = loc_data.flatten()
            
        cov_data = None
        scale_data = None
        
        if self.covariance_matrix is not None:
            cov_data = _extract_data(self.covariance_matrix)
            if cov_data.ndim != 2:
                d = loc_data.shape[0]
                cov_data = cov_data.reshape(d, d)
                
        if self.scale_tril is not None:
            scale_data = _extract_data(self.scale_tril)
            if scale_data.ndim != 2:
                d = loc_data.shape[0]
                scale_data = scale_data.reshape(d, d)
                
        return loc_data, cov_data, scale_data
    
    def forward(self, loc=None, cov=None, scale=None):

        # if self._distribution is None:
        #     loc, cov, scale = self._get_params()
        #     if scale is not None:
        #         self._distribution = dist.MultivariateNormal(loc, scale_tril=scale)
        #     else:
        #         print(cov)
        #         self._distribution = dist.MultivariateNormal(loc, covariance_matrix=cov)
        print(cov)
        if loc is None:
            loc = self.loc.data
        return dist.MultivariateNormal(loc = loc, covariance_matrix=cov)
    
    def mean(self):
        """Returns the mean. If loc is a quimb.Tensor, returns it directly."""
        return self.loc
    
    def entropy(self):
        return self.forward().entropy()
    
    def expected_log(self, loc=None, covariance_matrix=None, scale_tril=None) -> torch.Tensor:
        use_loc = loc if loc is not None else self.loc
        use_cov = covariance_matrix if covariance_matrix is not None else self.covariance_matrix
        use_scale = scale_tril if scale_tril is not None else self.scale_tril
        temp_dist = MultivariateGaussianDistribution(use_loc, use_cov, use_scale)
        return -temp_dist.entropy()

    def expected_log_prob(self, loc_p, covariance_matrix_p=None, scale_tril_p=None) -> torch.Tensor:
        # Note: Gaussian expected log prob is a scalar (summed over all dims), 
        # so we return a torch scalar, not a quimb tensor.
        mu_q, _, _ = self._get_params()
        d = mu_q.shape[0]
        
        if self.scale_tril is not None:
            scale = _extract_data(self.scale_tril).reshape(d, d)
            sigma_q = scale @ scale.T
        else:
            sigma_q = _extract_data(self.covariance_matrix).reshape(d, d)
            
        mu_p = _extract_data(loc_p).flatten()
        
        if scale_tril_p is not None:
            scale_p = _extract_data(scale_tril_p).reshape(d, d)
            sigma_p = scale_p @ scale_p.T
        elif covariance_matrix_p is not None:
            sigma_p = _extract_data(covariance_matrix_p).reshape(d, d)
        else:
            raise ValueError("Must provide either covariance_matrix_p or scale_tril_p")
            
        sigma_p_inv = torch.linalg.inv(sigma_p)
        sign, logdet_sigma_p = torch.linalg.slogdet(sigma_p)
        
        trace_term = torch.trace(sigma_p_inv @ sigma_q)
        diff = mu_q - mu_p
        mahalanobis_term = diff @ sigma_p_inv @ diff
        
        result = (-0.5 * d * torch.log(torch.tensor(2 * torch.pi, dtype=mu_q.dtype, device=mu_q.device))
                 - 0.5 * logdet_sigma_p
                 - 0.5 * (trace_term + mahalanobis_term))
        
        return result


class ProductDistribution:
    """Product of independent probability distributions."""
    def __init__(self, distributions: List[Union[GammaDistribution, MultivariateGaussianDistribution, 'ProductDistribution']]):
        self.distributions = distributions
    
    def forward(self):
        return [d.forward() for d in self.distributions]
    
    def mean(self):
        return [d.mean() for d in self.distributions]
    
    def entropy(self):
        # We need to handle potential quimb tensors in the sum
        entropies = []
        for d in self.distributions:
            ent = d.entropy()
            if isinstance(ent, qt.Tensor):
                entropies.append(ent.data) # extract torch tensor
            else:
                entropies.append(ent)
                
        total = torch.tensor(0.0, dtype=entropies[0].dtype, device=entropies[0].device)
        for ent in entropies:
            total = total + ent.sum()
        return total
    
    def sample(self, sample_shape=torch.Size()):
        samples = []
        for d in self.distributions:
            d_forward = d.forward()
            if isinstance(d_forward, list):
                samples.extend([dist_obj.sample(sample_shape) for dist_obj in d_forward]) # type: ignore
            else:
                samples.append(d_forward.sample(sample_shape)) # type: ignore
        return samples
    
    def log_prob(self, values: List):
        if len(values) != len(self.distributions):
            raise ValueError("Number of values must match number of distributions")
        log_probs = []
        for d, v in zip(self.distributions, values):
            d_forward = d.forward()
            if isinstance(d_forward, list):
                raise NotImplementedError("log_prob not implemented for nested ProductDistribution")
            else:
                log_probs.append(d_forward.log_prob(v)) # type: ignore
        return sum(log_probs)
    
    def expected_log(self, **kwargs) -> Any:
        # returns list of expectations (potentially quimb tensors)
        expected_logs = []
        for d in self.distributions:
            if hasattr(d, 'expected_log'):
                expected_logs.append(d.expected_log(**kwargs))
            else:
                raise NotImplementedError(f"Distribution {type(d)} does not implement expected_log")
        return expected_logs 

    def expected_log_prob(self, distributions_p, **kwargs) -> Any:
        # returns list of expectations (potentially quimb tensors)
        if len(distributions_p) != len(self.distributions):
            raise ValueError("Number of distributions must match")
        expected_log_probs = []
        for q_dist, p_dist in zip(self.distributions, distributions_p):
            if hasattr(q_dist, 'expected_log_prob'):
                if isinstance(p_dist, GammaDistribution):
                    elp = q_dist.expected_log_prob(concentration_p=p_dist.concentration, rate_p=p_dist.rate)
                elif isinstance(p_dist, MultivariateGaussianDistribution):
                    elp = q_dist.expected_log_prob(loc_p=p_dist.loc, covariance_matrix_p=p_dist.covariance_matrix, scale_tril_p=p_dist.scale_tril)
                elif isinstance(p_dist, ProductDistribution):
                    elp = q_dist.expected_log_prob(distributions_p=p_dist.distributions)
                else:
                    raise NotImplementedError(f"expected_log_prob not implemented for {type(p_dist)}")
                expected_log_probs.append(elp)
            else:
                raise NotImplementedError(f"Distribution {type(q_dist)} does not implement expected_log_prob")
        return expected_log_probs
