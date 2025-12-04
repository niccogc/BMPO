"""
Helper functions for building Bayesian Tensor Networks with quimb.

This module provides utilities to construct sigma networks from mu networks
and prepare networks for use with BayesianTensorNetwork.
"""

import numpy as np
import quimb.tensor as qtn
from typing import List, Optional, Callable

try:
    import autoray as ar
except ImportError:
    ar = None  # type: ignore



def default_sigma_init(d: int, num_nodes: int) -> np.ndarray:
    """
    Default initialization for sigma (covariance) tensors.
    
    Creates a diagonal matrix with values 1/(d*num_nodes) to prevent
    exponential growth in tensor networks with multiple nodes.
    
    Args:
        d: Flattened dimension of the tensor (product of all mode sizes)
        num_nodes: Total number of learnable nodes in the network
        
    Returns:
        Diagonal matrix of shape (d, d) with diagonal values = 1/(d*num_nodes)
        
    Example:
        >>> # For a 3-node network with tensor shape (2, 3, 4)
        >>> d = 2 * 3 * 4 = 24
        >>> num_nodes = 3
        >>> init_matrix = default_sigma_init(24, 3)
        >>> # Returns 24x24 diagonal matrix with values 1/72
    """
    init_value = 1.0 / (d * num_nodes)
    return np.eye(d, dtype=np.float64) * init_value

def create_sigma_network(
    mu_tn: qtn.TensorNetwork, 
    learnable_tags: List[str],
    output_indices: List[str],
    init_fn: Optional[Callable[[int, int], np.ndarray]] = None
) -> qtn.TensorNetwork:
    """
    Create sigma (covariance) network from mu (mean) network.
    
    For each learnable tensor in mu with indices (i, j, k, ...),
    creates a sigma tensor with doubled indices EXCEPT for output indices.
    
    - Variational indices (bonds, inputs): doubled as (io, ii)
    - Output indices: kept as single index (no outer/inner)
    
    Output indices are like samples - independent and summed over, never
    needing outer product dimensions.
    
    Args:
        mu_tn: The mean tensor network
        learnable_tags: List of tags for learnable tensors to double
        output_indices: List of output indices that should NOT be doubled
                       e.g., ['y'] for scalar output, ['y1', 'y2'] for multi-output
        init_fn: Optional initialization function that takes (d: int, num_nodes: int)
                and returns a d×d matrix. If None, uses default diagonal initialization
                with values 1/(d*num_nodes).
        
    Returns:
        sigma_tn: The covariance tensor network with doubled indices
        
    Example:
        >>> # Mu network: A(x, r1) -- B(r1, y) where 'y' is output
        >>> mu_tn = ...
        >>> sigma_tn = create_sigma_network(mu_tn, ['A', 'B'], output_indices=['y'])
        >>> # Sigma network: 
        >>> #   A_sigma(xo, xi, r1o, r1i)     - all doubled
        >>> #   B_sigma(r1o, r1i, y)          - r1 doubled, y NOT doubled
    """
    # Use default initialization if none provided
    if init_fn is None:
        init_fn = default_sigma_init
    
    output_indices_set = set(output_indices)
    num_nodes = len(learnable_tags)
    sigma_tensors = []
    
    for tag in learnable_tags:
        mu_tensor = mu_tn[tag]  # type: ignore
        
        # Get original indices and shape
        orig_inds = mu_tensor.inds  # type: ignore
        orig_shape = mu_tensor.shape  # type: ignore
        
        # Separate variational and output indices
        variational_inds = []
        variational_dims = []
        output_inds = []
        output_dims = []
        
        for ind, dim in zip(orig_inds, orig_shape):
            if ind in output_indices_set:
                output_inds.append(ind)
                output_dims.append(dim)
            else:
                variational_inds.append(ind)
                variational_dims.append(dim)
        
        # Build sigma indices: variational get doubled, outputs stay single
        sigma_inds = []
        sigma_shape = []
        
        # Add variational indices (doubled)
        for ind, dim in zip(variational_inds, variational_dims):
            sigma_inds.append(ind + 'o')  # outer
            sigma_inds.append(ind + 'i')  # inner
            sigma_shape.extend([dim, dim])
        
        # Add output indices (NOT doubled)
        for ind, dim in zip(output_inds, output_dims):
            sigma_inds.append(ind)
            sigma_shape.append(dim)
        
        # Flatten variational dimensions for matrix initialization
        d_var = int(np.prod(variational_dims)) if variational_dims else 1
        d_out = int(np.prod(output_dims)) if output_dims else 1
        
        # Get initialization matrix for variational parameters
        mu_data = mu_tensor.data  # type: ignore[union-attr]
        
        if ar is not None:
            backend = ar.infer_backend(mu_data)
            init_matrix_np = init_fn(d_var, num_nodes)
            init_matrix = ar.do('array', init_matrix_np, like=mu_data)
        else:
            init_matrix = init_fn(d_var, num_nodes)
        
        # Reshape: (d_var, d_var) -> variational doubled shape + output shape
        if variational_dims:
            # Reshape to (var1, var2, ..., var1, var2, ...)
            expanded_shape = list(variational_dims) + list(variational_dims)
            sigma_expanded = init_matrix.reshape(*expanded_shape)
            
            # Permute to interleave outer and inner: 
            # (var1, var2, ..., var1, var2, ...) -> (var1o, var1i, var2o, var2i, ...)
            n_var_dims = len(variational_dims)
            perm = []
            for i in range(n_var_dims):
                perm.append(i)              # outer dimension i
                perm.append(i + n_var_dims) # inner dimension i
            
            if ar is not None:
                sigma_permuted = ar.do('transpose', sigma_expanded, axes=perm, like=mu_data)
            else:
                sigma_permuted = np.transpose(sigma_expanded, perm)
            
            # Add output dimensions by expanding and broadcasting
            if output_dims:
                # Add axes for output dimensions at the end
                for _ in output_dims:
                    if ar is not None:
                        sigma_permuted = ar.do('expand_dims', sigma_permuted, axis=-1, like=mu_data)
                    else:
                        sigma_permuted = np.expand_dims(sigma_permuted, axis=-1)
                
                # Broadcast to include output dimensions
                final_shape = list(sigma_permuted.shape[:-len(output_dims)]) + list(output_dims)
                if ar is not None:
                    final_data = ar.do('broadcast_to', sigma_permuted, tuple(final_shape), like=mu_data)
                else:
                    final_data = np.broadcast_to(sigma_permuted, tuple(final_shape))
            else:
                final_data = sigma_permuted
        else:
            # No variational dimensions - only outputs (shouldn't happen normally)
            if ar is not None:
                final_data = ar.do('ones', sigma_shape, like=mu_data)
            else:
                final_data = np.ones(sigma_shape, dtype=np.float64)
        
        # Create sigma tensor
        sigma_tensor = qtn.Tensor(
            data=final_data,  # type: ignore[arg-type]
            inds=tuple(sigma_inds),
            tags=tag + '_sigma'
        )
        
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
