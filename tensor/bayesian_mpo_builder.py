"""
Builder functions for creating Bayesian MPO structures.

Provides utilities to construct μ-MPO and Σ-MPO with input blocks
following the Tensor Train (TT/MPS) pattern.
"""

import torch
from tensor.node import TensorNode
from tensor.bayesian_mpo import BayesianMPO
from typing import Tuple, Optional, List


def create_bayesian_tensor_train(
    num_blocks: int, 
    bond_dim: int, 
    input_features: int, 
    output_shape: int = 1,
    constrict_bond: bool = True,
    tau_alpha: Optional[torch.Tensor] = None,
    tau_beta: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None
) -> BayesianMPO:
    """
    Create a Bayesian MPO with Tensor Train (TT/MPS) structure.
    
    Structure:
    - μ-MPO: Standard TT with input blocks
    - Σ-MPO: Doubled TT with doubled input blocks (one for 'o', one for 'i')
    
    Args:
        num_blocks: Number of blocks/carriages (N)
        bond_dim: Bond dimension (r)
        input_features: Number of input features (f)
        output_shape: Output dimension (default: 1)
        constrict_bond: Whether to constrict bond dimensions
        tau_alpha: Alpha parameter for τ distribution
        tau_beta: Beta parameter for τ distribution
        dtype: Data type
        seed: Random seed
        
    Returns:
        BayesianMPO object with TT structure
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    dtype = dtype or torch.float64
    
    # Build μ-MPO nodes
    mu_nodes, mu_input_nodes = _build_mu_tensor_train(
        num_blocks, bond_dim, input_features, output_shape,
        constrict_bond, dtype
    )
    
    # Create BayesianMPO (this will automatically create Σ-MPO)
    bmpo = BayesianMPO(
        mu_nodes=mu_nodes,
        input_nodes=mu_input_nodes,
        tau_alpha=tau_alpha,
        tau_beta=tau_beta
    )
    
    # Create Σ-MPO input nodes (doubled)
    sigma_input_nodes = _build_sigma_input_nodes(mu_input_nodes, dtype)
    
    # Connect Σ-MPO input nodes to Σ-MPO main nodes
    _connect_sigma_inputs(sigma_input_nodes, bmpo.sigma_nodes)
    
    # Update BayesianMPO with Σ input nodes
    bmpo.sigma_input_nodes = sigma_input_nodes  # type: ignore
    bmpo.sigma_mpo.input_nodes = sigma_input_nodes
    
    # Re-discover nodes to include input nodes in the network
    bmpo.sigma_mpo.nodes, bmpo.sigma_mpo.node_indices = bmpo.sigma_mpo._discover_nodes()
    
    return bmpo


def _build_mu_tensor_train(
    num_blocks: int, 
    bond_dim: int, 
    input_features: int, 
    output_shape: int,
    constrict_bond: bool, 
    dtype: torch.dtype
) -> Tuple[List[TensorNode], List[TensorNode]]:
    """
    Build μ-MPO nodes following Tensor Train pattern.
    
    Returns:
        mu_nodes: List of main nodes
        mu_input_nodes: List of input nodes
    """
    # Build bond dimensions
    bond_dims = _compute_bond_dimensions(num_blocks, bond_dim, input_features,
                                         constrict_bond)
    
    # Create main nodes
    mu_nodes = []
    for i in range(num_blocks):
        left_dim = bond_dims[i]
        right_dim = bond_dims[i + 1]
        
        # Output dimension (only last block has output if output_shape > 1)
        if i == num_blocks - 1 and output_shape > 1:
            out_dim = output_shape
        else:
            out_dim = 1
        
        # Shape: (left_bond, output, feature, right_bond)
        shape = (left_dim, out_dim, input_features, right_dim)
        labels = [f'r{i}', f'c{i+1}', f'p{i+1}', f'r{i+1}']
        
        node = TensorNode(
            tensor_or_shape=shape,
            dim_labels=labels,
            l=f'r{i}',
            r=f'r{i+1}',
            name=f'mu{i+1}',
            dtype=dtype
        )
        
        # Squeeze output dimension if it's 1
        if out_dim == 1:
            node.squeeze(exclude={f'c{i+1}'})
        
        mu_nodes.append(node)
    
    # Connect nodes horizontally
    for i in range(len(mu_nodes) - 1):
        mu_nodes[i].connect(mu_nodes[i + 1], f'r{i+1}', priority=1)
    
    # Create input nodes
    mu_input_nodes = []
    for i in range(num_blocks):
        # Shape: (sample, feature)
        input_node = TensorNode(
            tensor_or_shape=(1, input_features),
            dim_labels=['s', f'p{i+1}'],
            name=f'X{i+1}',
            dtype=dtype
        )
        mu_input_nodes.append(input_node)
    
    # Connect input nodes to main nodes
    for i in range(num_blocks):
        mu_input_nodes[i].connect(mu_nodes[i], f'p{i+1}')
    
    return mu_nodes, mu_input_nodes


def _build_sigma_input_nodes(mu_input_nodes: List[TensorNode], dtype: torch.dtype) -> List[TensorNode]:
    """
    Build Σ-MPO input nodes (doubled).
    
    For each μ input node with shape (1, f) and labels ['s', 'p1'],
    create two Σ input nodes:
    - One with labels ['s', 'p1o'] (outer)
    - One with labels ['s', 'p1i'] (inner)
    
    Returns:
        List of Σ input nodes (2 * len(mu_input_nodes))
    """
    sigma_input_nodes = []
    
    for mu_input in mu_input_nodes:
        # Extract feature dimension and label
        feature_dim = mu_input.shape[1]
        # Get the 'p' label (e.g., 'p1', 'p2', etc.)
        p_label = [l for l in mu_input.dim_labels if l.startswith('p')][0]
        
        # Create outer input node
        input_node_o = TensorNode(
            tensor_or_shape=(1, feature_dim),
            dim_labels=['s', f'{p_label}o'],
            name=f'{mu_input.name}_o',
            dtype=dtype
        )
        
        # Create inner input node
        input_node_i = TensorNode(
            tensor_or_shape=(1, feature_dim),
            dim_labels=['s', f'{p_label}i'],
            name=f'{mu_input.name}_i',
            dtype=dtype
        )
        
        sigma_input_nodes.extend([input_node_o, input_node_i])
    
    return sigma_input_nodes


def _connect_sigma_inputs(sigma_input_nodes: List[TensorNode], sigma_main_nodes: List[TensorNode]) -> None:
    """
    Connect Σ input nodes to Σ main nodes.
    
    For each Σ main node, connect:
    - One input node to 'o' (outer) feature dimension
    - One input node to 'i' (inner) feature dimension
    """
    # Group input nodes in pairs (outer, inner)
    for i in range(len(sigma_main_nodes)):
        sigma_node = sigma_main_nodes[i]
        
        # Get the 'o' and 'i' labels for features (e.g., 'p1o', 'p1i')
        fo_label = [l for l in sigma_node.dim_labels if l.startswith('p') and l.endswith('o')][0]
        fi_label = [l for l in sigma_node.dim_labels if l.startswith('p') and l.endswith('i')][0]
        
        # Connect corresponding input nodes
        input_o = sigma_input_nodes[2 * i]      # Outer input
        input_i = sigma_input_nodes[2 * i + 1]  # Inner input
        
        input_o.connect(sigma_node, fo_label)
        input_i.connect(sigma_node, fi_label)


def _compute_bond_dimensions(num_blocks: int, bond_dim: int, input_features: int, constrict_bond: bool) -> List[int]:
    """
    Compute bond dimensions for each connection.
    
    Returns:
        List of bond dimensions [1, r1, r2, ..., rN, 1]
    """
    if num_blocks == 1:
        return [1, 1]
    
    bond_dims = [1]  # Left boundary
    
    for i in range(num_blocks - 1):
        if constrict_bond:
            # Grow from left
            if i < num_blocks // 2:
                left_dim = bond_dims[-1]
                max_dim = min(bond_dim, left_dim * input_features)
                bond_dims.append(max_dim)
            # Shrink from right
            else:
                # Will be adjusted later
                bond_dims.append(bond_dim)
        else:
            bond_dims.append(bond_dim)
    
    bond_dims.append(1)  # Right boundary
    
    # Adjust right side if constricting
    if constrict_bond and num_blocks > 2:
        for i in range(num_blocks - 1, num_blocks // 2, -1):
            right_dim = bond_dims[i + 1]
            max_dim = min(bond_dim, right_dim * input_features)
            bond_dims[i] = min(bond_dims[i], max_dim)
    
    return bond_dims
