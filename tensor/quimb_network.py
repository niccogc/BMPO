"""
Quimb-based Tensor Network for Bayesian Inference

This module provides a wrapper around quimb's TensorNetwork
for use in Bayesian variational inference, with PyTorch integration.

Heavily based on BMPONetwork structure.
"""

import torch
import quimb.tensor as qtn
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from collections import defaultdict

from .probability_distributions import GammaDistribution, ProductDistribution


class QuimbTensorNetwork:
    """
    Wrapper for quimb TensorNetwork that manages Gamma distributions over bonds.
    
    Similar to BMPONetwork but for arbitrary tensor network topologies using quimb.
    """
    
    def __init__(
        self,
        tn: qtn.TensorNetwork,
        learnable_tags: List[str],
        input_tags: Optional[List[str]] = None,
        distributions: Optional[Dict[str, Dict[str, Any]]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize QuimbTensorNetwork.
        
        Args:
            tn: Quimb TensorNetwork (mu network)
            learnable_tags: List of tensor tags that are learnable nodes
            input_tags: List of tensor tags that are input nodes (fixed)
            distributions: Dict of distribution parameters per bond (index label)
                          If None, initialized with default values
            device: PyTorch device
            dtype: PyTorch dtype
        """
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # Store the mu network
        self.mu_tn = tn.copy()
        
        # Node organization
        self.learnable_tags = list(learnable_tags)
        self.input_tags = list(input_tags) if input_tags else []
        
        # Identify bonds (shared indices between tensors)
        self.bond_labels = self._identify_bonds()
        
        # Initialize distributions
        self.distributions: Dict[str, Dict[str, Any]]
        if distributions is None:
            self.distributions = {}
            self._initialize_distributions()
        else:
            self.distributions = distributions
        
        # Build mapping from bonds to associated nodes
        self.bond_to_nodes = self._build_bond_to_nodes_mapping()
    
    def _identify_bonds(self) -> List[str]:
        """
        Identify all bonds (shared indices) in the network.
        
        Returns:
            List of bond names (index labels that appear in multiple tensors)
        """
        bonds: Set[str] = set()
        
        # Count occurrences of each index
        ind_count: Dict[str, int] = defaultdict(int)
        ind_sizes: Dict[str, int] = {}
        
        for tensor in self.mu_tn.tensor_map.values():
            for ind, size in zip(tensor.inds, tensor.shape):
                ind_count[ind] += 1
                ind_sizes[ind] = size
        
        # Bonds are indices that appear in multiple tensors
        for ind, count in ind_count.items():
            if count > 1:
                bonds.add(ind)
        
        return sorted(list(bonds))
    
    def _initialize_distributions(self) -> None:
        """
        Initialize Gamma distributions for each bond.
        
        Similar to BMPONetwork._initialize_distributions()
        """
        for bond in self.bond_labels:
            # Get bond dimension
            bond_dim = self._get_bond_dimension(bond)
            
            # Initialize with weak prior: Gamma(1, 0.1)
            alpha_0 = torch.ones(bond_dim, device=self.device, dtype=self.dtype)
            beta_0 = torch.ones(bond_dim, device=self.device, dtype=self.dtype) * 0.1
            
            self.distributions[bond] = {
                'alpha': alpha_0.clone(),
                'beta': beta_0.clone(),
                'alpha_0': alpha_0.clone(),
                'beta_0': beta_0.clone(),
                'expectation': alpha_0 / beta_0,
                'distribution': GammaDistribution(concentration=alpha_0.clone(), rate=beta_0.clone())
            }
    
    def _get_bond_dimension(self, bond: str) -> int:
        """Get dimension of a bond (shared index)."""
        for tensor in self.mu_tn.tensor_map.values():
            if bond in tensor.inds:
                idx = tensor.inds.index(bond)
                return tensor.shape[idx]
        raise ValueError(f"Bond {bond} not found in network")
    
    def _build_bond_to_nodes_mapping(self) -> Dict[str, List[str]]:
        """
        Build mapping from bond names to connected node tags.
        
        Returns:
            Dict mapping bond label to list of learnable node tags
        """
        bond_to_nodes: Dict[str, List[str]] = defaultdict(list)
        
        for tid, tensor in self.mu_tn.tensor_map.items():
            # Get the tag for this tensor
            tag = list(tensor.tags)[0] if tensor.tags else tid
            
            # Only consider learnable nodes
            if tag not in self.learnable_tags:
                continue
            
            for ind in tensor.inds:
                if ind in self.bond_labels:
                    bond_to_nodes[ind].append(tag)
        
        return dict(bond_to_nodes)
    
    def get_distribution_params(self, bond: str) -> Dict[str, Any]:
        """Get distribution parameters for a bond."""
        return self.distributions[bond]
    
    def get_expectations(self) -> Dict[str, torch.Tensor]:
        """Get expectations for all bonds."""
        expectations: Dict[str, torch.Tensor] = {}
        for bond in self.bond_labels:
            exp = self.distributions[bond]['expectation']
            if isinstance(exp, torch.Tensor):
                expectations[bond] = exp
        return expectations
    
    def update_distribution_params(self, bond: str, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        """
        Update distribution parameters for a bond.
        
        Similar to BMPONetwork.update_distribution_params()
        """
        self.distributions[bond]['alpha'] = alpha
        self.distributions[bond]['beta'] = beta
        dist = self.distributions[bond]['distribution']
        if isinstance(dist, GammaDistribution):
            dist.update_parameters(alpha, beta)
        self._update_expectations()
    
    def _update_expectations(self) -> None:
        """Update expectations for all distributions."""
        for bond in self.bond_labels:
            alpha = self.distributions[bond]['alpha']
            beta = self.distributions[bond]['beta']
            if isinstance(alpha, torch.Tensor) and isinstance(beta, torch.Tensor):
                self.distributions[bond]['expectation'] = alpha / beta
    
    def get_gamma_distributions(self) -> Dict[str, GammaDistribution]:
        """Get Gamma distribution objects for all bonds."""
        gamma_dists: Dict[str, GammaDistribution] = {}
        for bond in self.bond_labels:
            dist = self.distributions[bond]['distribution']
            if isinstance(dist, GammaDistribution):
                gamma_dists[bond] = dist
        return gamma_dists
    
    def get_product_distribution(self) -> ProductDistribution:
        """Get product distribution over all bonds."""
        dists = []
        for bond in self.bond_labels:
            dist = self.distributions[bond]['distribution']
            if isinstance(dist, GammaDistribution):
                dists.append(dist)
        return ProductDistribution(dists)
    
    def get_nodes_for_bond(self, bond: str) -> List[str]:
        """
        Get list of learnable node tags connected to a bond.
        
        Similar to BMPONetwork.get_nodes_for_bond()
        """
        return self.bond_to_nodes.get(bond, [])
    
    def compute_theta_tensor(self, node_tag: str, exclude_labels: Optional[List[str]] = None) -> torch.Tensor:
        """
        Compute the Theta tensor for a node from expectations of Gamma distributions.
        
        Theta is the tensor product of E_q[λ] for each bond in the node.
        
        Similar to BMPONetwork.compute_theta_tensor() but for arbitrary node.
        
        Args:
            node_tag: Tag identifying the node
            exclude_labels: List of bond labels to exclude (set their dimension to 1)
            
        Returns:
            Theta tensor with shape matching the node shape
        """
        if exclude_labels is None:
            exclude_labels = []
        
        # Get the node tensor
        node_tensor = self.mu_tn[node_tag]  # type: ignore
        
        # Collect factors (expectation vectors) for each dimension
        factors = []
        for ind, size in zip(node_tensor.inds, node_tensor.shape):  # type: ignore
            if ind in exclude_labels:
                # Excluded: use vector of ones with size 1
                factor = torch.ones(1, dtype=self.dtype, device=self.device)
            else:
                # Get expectations for this bond
                if ind in self.distributions:
                    exp = self.distributions[ind]['expectation']
                    if isinstance(exp, torch.Tensor):
                        factor = exp
                    else:
                        factor = torch.ones(size, dtype=self.dtype, device=self.device)
                else:
                    # Not a bond (e.g., input dimension): use ones
                    factor = torch.ones(size, dtype=self.dtype, device=self.device)
            factors.append(factor)
        
        # Use einsum for efficient outer product
        num_dims = len(factors)
        
        if num_dims == 0:
            return torch.tensor(1.0, dtype=self.dtype, device=self.device)
        elif num_dims == 1:
            factor_0 = factors[0]
            assert isinstance(factor_0, torch.Tensor)
            return factor_0
        else:
            # Multiple dimensions: use einsum
            # Create symbolic indices for each dimension
            input_indices = ','.join([chr(97 + i) for i in range(num_dims)])  # 'a,b,c,...'
            output_indices = ''.join([chr(97 + i) for i in range(num_dims)])   # 'abc...'
            einsum_str = f'{input_indices}->{output_indices}'
            
            theta = torch.einsum(einsum_str, *factors)
        
        return theta
    
    def compute_entropy(self) -> torch.Tensor:
        """
        Compute entropy of the product distribution over all bonds.
        
        Similar to BMPONetwork.compute_entropy()
        """
        return self.get_product_distribution().entropy()  # type: ignore
    
    def get_node_tensor(self, node_tag: str) -> torch.Tensor:
        """Get tensor data for a node as PyTorch tensor."""
        tensor = self.mu_tn[node_tag]  # type: ignore
        data = tensor.data  # type: ignore
        
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data.copy()).to(device=self.device, dtype=self.dtype)
        return torch.tensor(data, device=self.device, dtype=self.dtype)
    
    def set_node_tensor(self, node_tag: str, data: torch.Tensor) -> None:
        """Set tensor data for a node."""
        data_np: np.ndarray
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.array(data)
        
        self.mu_tn[node_tag].modify(data=data_np)  # type: ignore
    
    def get_node_shape(self, node_tag: str) -> Tuple[int, ...]:
        """Get shape of a node tensor."""
        return tuple(self.mu_tn[node_tag].shape)  # type: ignore
    
    def get_node_inds(self, node_tag: str) -> Tuple[str, ...]:
        """Get index labels for a node."""
        return self.mu_tn[node_tag].inds  # type: ignore
    
    def to(self, device: torch.device) -> 'QuimbTensorNetwork':
        """Move all distributions to device."""
        self.device = device
        for bond in self.bond_labels:
            dist_dict = self.distributions[bond]
            for key in ['alpha', 'beta', 'alpha_0', 'beta_0', 'expectation']:
                val = dist_dict[key]
                if isinstance(val, torch.Tensor):
                    dist_dict[key] = val.to(device)
            
            # Distribution objects handle their own device management via update_parameters
            # So we update them with their current values on the new device
            dist = dist_dict['distribution']
            if isinstance(dist, GammaDistribution):
                alpha_dev = dist_dict['alpha']
                beta_dev = dist_dict['beta']
                if isinstance(alpha_dev, torch.Tensor) and isinstance(beta_dev, torch.Tensor):
                    dist.update_parameters(alpha_dev, beta_dev)
        return self
