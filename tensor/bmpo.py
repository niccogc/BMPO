"""
Bayesian Matrix Product Operator (BMPO) module.

Each node has distributions for each mode:
- Rank dimensions (horizontal bonds): Gamma(c, e) with variable ω
- Vertical dimensions: Gamma(f, g) with variable φ
- Each index in a dimension has its own distribution

For a dimension of size N, there are N Gamma distributions (one per index).
"""

import torch
from tensor.node import TensorNode
from tensor.network import TensorNetwork
from tensor.probability_distributions import GammaDistribution, ProductDistribution
from typing import Dict, List, Optional


class BMPONetwork(TensorNetwork):
    """
    BMPO Network for Bayesian tensor networks with Gamma distributions per mode.
    
    Each dimension label has associated Gamma distribution parameters:
    - Rank dimensions (in left_labels/right_labels): variable ω, params (c, e)
    - Vertical dimensions: variable φ, params (f, g)
    
    For a dimension of size N, stores N sets of Gamma parameters (one per index).
    
    Structure:
        distributions[label] = {
            'type': 'rank' or 'vertical',
            'variable': 'omega' or 'phi',
            'c' or 'f': tensor of length N (first Gamma parameter),
            'e' or 'g': tensor of length N (second Gamma parameter),
            'expectation': tensor of length N (E[ω] or E[φ])
        }
    """
    
    def __init__(self, input_nodes, main_nodes, train_nodes=None, 
                 output_labels=('s',), sample_dim='s', 
                 rank_labels=None, distributions=None):
        """
        Initialize BMPO Network.
        
        Args:
            input_nodes: List of input nodes
            main_nodes: List of main nodes
            train_nodes: List of trainable nodes (default: main_nodes)
            output_labels: Labels for output dimensions
            sample_dim: Sample dimension label
            rank_labels: Set of dimension labels that are rank dimensions (horizontal bonds)
                        If None, inferred from node left_labels and right_labels
            distributions: Dict of distribution parameters per dimension label
                          If None, initialized with default values
        """
        super().__init__(input_nodes, main_nodes, train_nodes, output_labels, sample_dim)
        
        # Determine which dimensions are ranks (horizontal bonds)
        if rank_labels is None:
            self.rank_labels = self._infer_rank_labels()
        else:
            self.rank_labels = set(rank_labels)
        
        # Initialize distributions
        if distributions is None:
            self.distributions = {}
            self._initialize_distributions()
        else:
            self.distributions = distributions
    
    def _infer_rank_labels(self):
        """Infer which dimensions are ranks from node left_labels and right_labels."""
        rank_labels = set()
        for node in self.main_nodes:
            rank_labels.update(node.left_labels)
            rank_labels.update(node.right_labels)
        return rank_labels
    
    def _initialize_distributions(self):
        """Initialize Gamma distributions for all unique dimension labels."""
        for node in self.main_nodes:
            for label, size in zip(node.dim_labels, node.shape):
                if label not in self.distributions:
                    dtype = node.tensor.dtype
                    
                    if label in self.rank_labels:
                        # Rank dimension: Gamma(c, e) with variable ω
                        self.distributions[label] = {
                            'type': 'rank',
                            'variable': 'omega',
                            'c': torch.ones(size, dtype=dtype) * 2.0,  # Default c
                            'e': torch.ones(size, dtype=dtype) * 1.0,  # Default e
                            'expectation': torch.ones(size, dtype=dtype) * 2.0  # E[ω] = c/e
                        }
                    else:
                        # Vertical dimension: Gamma(f, g) with variable φ
                        self.distributions[label] = {
                            'type': 'vertical',
                            'variable': 'phi',
                            'f': torch.ones(size, dtype=dtype) * 2.0,  # Default f
                            'g': torch.ones(size, dtype=dtype) * 1.0,  # Default g
                            'expectation': torch.ones(size, dtype=dtype) * 2.0  # E[φ] = f/g
                        }
    
    def get_distribution_params(self, label):
        """Get distribution parameters for a dimension label."""
        return self.distributions.get(label)
    
    def get_expectations(self, label):
        """Get expectation values E[ω] or E[φ] for a dimension."""
        dist = self.distributions.get(label)
        return dist['expectation'] if dist else None
    
    def update_distribution_params(self, label, param1=None, param2=None):
        """
        Update Gamma distribution parameters for a dimension.
        
        Args:
            label: Dimension label
            param1: For rank: c values; For vertical: f values
            param2: For rank: e values; For vertical: g values
        """
        if label not in self.distributions:
            raise ValueError(f"Label '{label}' not found in distributions")
        
        dist = self.distributions[label]
        
        if dist['type'] == 'rank':
            if param1 is not None:
                dist['c'] = param1
            if param2 is not None:
                dist['e'] = param2
            # Update expectation using mean() from GammaDistribution objects
            self._update_expectations(label)
        else:  # vertical
            if param1 is not None:
                dist['f'] = param1
            if param2 is not None:
                dist['g'] = param2
            # Update expectation using mean() from GammaDistribution objects
            self._update_expectations(label)
    
    def _update_expectations(self, label):
        """Update expectation values from Gamma distribution means."""
        gammas = self.get_gamma_distributions(label)
        expectations = torch.tensor([gamma.mean() for gamma in gammas])
        self.distributions[label]['expectation'] = expectations
    
    def get_gamma_distributions(self, label):
        """
        Get list of Gamma distribution objects for a dimension.
        
        Args:
            label: Dimension label
            
        Returns:
            List of GammaDistribution objects, one per index in dimension
        """
        dist = self.distributions.get(label)
        if dist is None:
            return None
        
        distributions = []
        if dist['type'] == 'rank':
            # Gamma(c, e) for each index
            for i in range(len(dist['c'])):
                gamma = GammaDistribution(
                    concentration=dist['c'][i],
                    rate=dist['e'][i]
                )
                distributions.append(gamma)
        else:  # vertical
            # Gamma(f, g) for each index
            for i in range(len(dist['f'])):
                gamma = GammaDistribution(
                    concentration=dist['f'][i],
                    rate=dist['g'][i]
                )
                distributions.append(gamma)
        
        return distributions
    
    def get_product_distribution(self, label):
        """
        Get product distribution for all indices in a dimension.
        
        Returns:
            ProductDistribution of all Gamma distributions for this dimension
        """
        gammas = self.get_gamma_distributions(label)
        if gammas is None:
            return None
        return ProductDistribution(gammas)
    
    def compute_entropy(self, label):
        """
        Compute total entropy for a dimension (sum of entropies).
        
        Args:
            label: Dimension label
            
        Returns:
            Total entropy (sum over all indices)
        """
        product_dist = self.get_product_distribution(label)
        if product_dist is None:
            return None
        return product_dist.entropy()
    
    def get_jacobian(self, node):
        """
        Compute Jacobian (J) for a node - the network contracted without that block.
        
        Args:
            node: Node to compute Jacobian for
            
        Returns:
            jacobian_node: TensorNode representing J
        """
        return self.compute_jacobian_stack(node)
    
    def trim(self, thresholds):
        """
        Trim all nodes in the network based on expectation value thresholds.
        
        For each dimension label, keeps only indices where E[ω] or E[φ] >= threshold.
        Trims tensors and distribution parameters accordingly.
        
        Args:
            thresholds: Dict mapping dim_label -> threshold value
        """
        keep_indices = {}
        
        for label, dist in self.distributions.items():
            threshold = thresholds.get(label, float('-inf'))
            expectations = dist['expectation']
            
            # Find indices where expectation >= threshold
            mask = expectations >= threshold
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                raise ValueError(
                    f"Trimming dimension '{label}' with threshold {threshold} "
                    f"would remove all indices. Max expectation: {expectations.max().item()}"
                )
            
            keep_indices[label] = indices
            
            # Trim distribution parameters
            if dist['type'] == 'rank':
                dist['c'] = dist['c'][indices]
                dist['e'] = dist['e'][indices]
            else:  # vertical
                dist['f'] = dist['f'][indices]
                dist['g'] = dist['g'][indices]
            dist['expectation'] = dist['expectation'][indices]
        
        # Trim all node tensors
        all_nodes = self.input_nodes + self.main_nodes
        for node in all_nodes:
            self._trim_node(node, keep_indices)
        
        # Reset stacks after trimming
        self.reset_stacks()
        
        return self
    
    def _trim_node(self, node, keep_indices):
        """Trim a node's tensor based on keep_indices for each dimension."""
        new_tensor = node.tensor
        
        for dim_idx, label in enumerate(node.dim_labels):
            if label in keep_indices:
                indices = keep_indices[label]
                new_tensor = torch.index_select(new_tensor, dim_idx, indices)
        
        node.tensor = new_tensor
    
    def to(self, device=None, dtype=None):
        """Move network tensors and distribution parameters to device/dtype."""
        super().to(device=device, dtype=dtype)
        for label, dist in self.distributions.items():
            if dist['type'] == 'rank':
                dist['c'] = dist['c'].to(device=device, dtype=dtype)
                dist['e'] = dist['e'].to(device=device, dtype=dtype)
            else:
                dist['f'] = dist['f'].to(device=device, dtype=dtype)
                dist['g'] = dist['g'].to(device=device, dtype=dtype)
            dist['expectation'] = dist['expectation'].to(device=device, dtype=dtype)
        return self
