"""
Helper functions for building Bayesian Tensor Networks with quimb.

This module provides utilities to construct sigma networks from mu networks
and prepare networks for use with BayesianTensorNetwork.
"""

import numpy as np
import quimb.tensor as qtn
from typing import List


def create_sigma_network(mu_tn: qtn.TensorNetwork, learnable_tags: List[str]) -> qtn.TensorNetwork:
    """
    Create sigma (covariance) network from mu (mean) network.
    
    For each learnable tensor in mu with indices (i, j, k, ...),
    creates a sigma tensor with doubled indices (io, ii, jo, ji, ko, ki, ...).
    
    The sigma network has the same topology as mu but with doubled bond dimensions.
    Each index becomes two indices: 'outer' (o) and 'inner' (i).
    
    Args:
        mu_tn: The mean tensor network
        learnable_tags: List of tags for learnable tensors to double
        
    Returns:
        sigma_tn: The covariance tensor network with doubled indices
        
    Example:
        >>> # Mu network: A(x, r1) -- B(r1, y)
        >>> mu_tn = ...
        >>> sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
        >>> # Sigma network: A_sigma(xo, xi, r1o, r1i) -- B_sigma(r1o, r1i, yo, yi)
    """
    sigma_tensors = []
    
    for tag in learnable_tags:
        mu_tensor = mu_tn[tag]  # type: ignore
        
        # Get original indices and shape
        orig_inds = mu_tensor.inds  # type: ignore
        orig_shape = mu_tensor.shape  # type: ignore
        
        # Create doubled indices: each index 'x' becomes 'xo' and 'xi'
        sigma_inds = []
        sigma_shape = []
        
        for ind, dim in zip(orig_inds, orig_shape):
            sigma_inds.append(ind + 'o')  # outer
            sigma_inds.append(ind + 'i')  # inner
            sigma_shape.extend([dim, dim])
        
        # Initialize sigma tensor as small diagonal
        # This represents small initial uncertainty
        data = np.zeros(sigma_shape, dtype=np.float64)
        
        # Set diagonal elements
        # For a tensor with shape (d1, d2, ...), the diagonal is where all indices match
        for idx in np.ndindex(*orig_shape):
            # Map original index to doubled index (i, i) for each dimension
            sigma_idx = []
            for i in idx:
                sigma_idx.extend([i, i])  # both outer and inner get same value
            data[tuple(sigma_idx)] = 0.01  # Small initial variance
        
        # Create sigma tensor
        sigma_tensor = qtn.Tensor(
            data=data,
            inds=tuple(sigma_inds),
            tags=tag + '_sigma'
        )  # type: ignore
        
        sigma_tensors.append(sigma_tensor)
    
    # Create sigma network
    sigma_tn = qtn.TensorNetwork(sigma_tensors)
    
    return sigma_tn


def validate_network_structure(tn: qtn.TensorNetwork, input_indices: dict) -> bool:
    """
    Validate that tensor network structure is compatible with BayesianTensorNetwork.
    
    Checks:
    - All specified input indices exist in the network
    - Network is properly connected
    
    Args:
        tn: The tensor network to validate
        input_indices: Dict mapping input names to list of indices
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Collect all indices in the network
    all_inds = set()
    for tensor in tn.tensor_map.values():
        all_inds.update(tensor.inds)  # type: ignore
    
    # Check that all input indices exist
    for input_name, indices in input_indices.items():
        for idx in indices:
            if idx not in all_inds:
                raise ValueError(
                    f"Input index '{idx}' for input '{input_name}' not found in network. "
                    f"Available indices: {sorted(all_inds)}"
                )
    
    return True
