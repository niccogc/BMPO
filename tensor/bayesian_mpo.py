"""
Bayesian Matrix Product Operator (Full Structure).

Contains:
- μ-MPO: Mean tensor network (standard MPO)
- Σ-MPO: Variation tensor network (doubled structure with 'o' and 'i' indices)
- Prior distributions: Gamma distributions for each mode of μ-MPO
- τ distribution: Gamma(α, β) for noise/regularization
"""

import torch
from tensor.node import TensorNode
from tensor.network import TensorNetwork
from tensor.bmpo import BMPONetwork
from tensor.probability_distributions import GammaDistribution
from typing import Dict, List, Optional, Tuple


class BayesianMPO:
    """
    Full Bayesian MPO structure.
    
    Structure:
        - μ-MPO: Standard tensor network with blocks like (r1, f1, r2)
        - Σ-MPO: Doubled tensor network with blocks like (r1o, r1i, f1o, f1i, r2o, r2i)
          where contractions happen between 'o' (outer) and 'i' (inner) separately
        - Prior parameters: Gamma distributions for μ-MPO modes
        - τ: Gamma(α, β) noise distribution
    
    Example:
        μ-block shape: (3, 4, 5) with labels ['r1', 'f1', 'r2']
        Σ-block shape: (3, 3, 4, 4, 5, 5) with labels ['r1o', 'r1i', 'f1o', 'f1i', 'r2o', 'r2i']
    """
    
    def __init__(self, mu_nodes, input_nodes=None, rank_labels=None, 
                 tau_alpha=None, tau_beta=None):
        """
        Initialize Bayesian MPO.
        
        Args:
            mu_nodes: List of nodes for μ-MPO (standard MPO blocks)
            input_nodes: List of input nodes (optional)
            rank_labels: Set of dimension labels that are ranks (horizontal bonds)
            tau_alpha: Alpha parameter for τ ~ Gamma(α, β)
            tau_beta: Beta parameter for τ ~ Gamma(α, β)
        """
        self.input_nodes = input_nodes or []
        self.mu_nodes = mu_nodes
        
        # Create μ-MPO network with prior distributions
        self.mu_mpo = BMPONetwork(
            input_nodes=self.input_nodes,
            main_nodes=self.mu_nodes,
            rank_labels=rank_labels
        )
        
        # Create Σ-MPO structure (doubled)
        self.sigma_nodes = self._create_sigma_nodes()
        self.sigma_mpo = TensorNetwork(
            input_nodes=[],  # Σ-MPO doesn't directly connect to inputs
            main_nodes=self.sigma_nodes
        )
        
        # τ distribution: Gamma(α, β)
        if tau_alpha is None or tau_beta is None:
            # Default values
            self.tau_alpha = torch.tensor(2.0)
            self.tau_beta = torch.tensor(1.0)
        else:
            self.tau_alpha = tau_alpha
            self.tau_beta = tau_beta
        
        self.tau_distribution = GammaDistribution(
            concentration=self.tau_alpha,
            rate=self.tau_beta
        )
    
    def _create_sigma_nodes(self):
        """
        Create Σ-MPO nodes from μ-MPO nodes.
        
        Each μ-node with shape (d1, d2, d3) and labels ['r1', 'f1', 'r2']
        becomes a Σ-node with shape (d1, d1, d2, d2, d3, d3) 
        and labels ['r1o', 'r1i', 'f1o', 'f1i', 'r2o', 'r2i']
        """
        sigma_nodes = []
        
        for mu_node in self.mu_nodes:
            # Double the shape: each dimension becomes (dim, dim)
            sigma_shape = []
            sigma_labels = []
            sigma_left_labels = []
            sigma_right_labels = []
            
            for dim_size, label in zip(mu_node.shape, mu_node.dim_labels):
                # Create 'o' (outer) and 'i' (inner) versions
                sigma_shape.extend([dim_size, dim_size])
                label_o = f"{label}o"
                label_i = f"{label}i"
                sigma_labels.extend([label_o, label_i])
                
                # Handle left/right bond labels
                if label in mu_node.left_labels:
                    sigma_left_labels.extend([label_o, label_i])
                if label in mu_node.right_labels:
                    sigma_right_labels.extend([label_o, label_i])
            
            # Create Σ-node
            sigma_node = TensorNode(
                tensor_or_shape=tuple(sigma_shape),
                dim_labels=sigma_labels,
                l=sigma_left_labels if sigma_left_labels else None,
                r=sigma_right_labels if sigma_right_labels else None,
                name=f"sigma_{mu_node.name}"
            )
            
            sigma_nodes.append(sigma_node)
        
        # Connect Σ-nodes (outer to outer, inner to inner)
        for i in range(len(sigma_nodes) - 1):
            curr_node = sigma_nodes[i]
            next_node = sigma_nodes[i + 1]
            
            # Find shared bond labels
            for label in curr_node.dim_labels:
                if label in next_node.dim_labels:
                    curr_node.connect(next_node, label)
        
        return sigma_nodes
    
    def get_mu_distributions(self, label):
        """Get Gamma distributions for a μ-MPO dimension."""
        return self.mu_mpo.get_gamma_distributions(label)
    
    def get_mu_expectations(self, label):
        """Get expectation values E[ω] or E[φ] for μ-MPO dimension."""
        return self.mu_mpo.get_expectations(label)
    
    def update_mu_params(self, label, param1=None, param2=None):
        """Update μ-MPO distribution parameters."""
        self.mu_mpo.update_distribution_params(label, param1, param2)
    
    def update_tau(self, alpha=None, beta=None):
        """
        Update τ distribution parameters.
        
        Args:
            alpha: New α parameter
            beta: New β parameter
        """
        if alpha is not None:
            self.tau_alpha = alpha
        if beta is not None:
            self.tau_beta = beta
        
        self.tau_distribution = GammaDistribution(
            concentration=self.tau_alpha,
            rate=self.tau_beta
        )
    
    def get_tau_mean(self):
        """Get mean of τ distribution: E[τ] = α/β."""
        return self.tau_distribution.mean()
    
    def get_tau_entropy(self):
        """Get entropy of τ distribution."""
        return self.tau_distribution.entropy()
    
    def forward_mu(self, x, to_tensor=False):
        """
        Forward pass through μ-MPO.
        
        Args:
            x: Input data
            to_tensor: If True, return tensor instead of TensorNode
            
        Returns:
            Output from μ-MPO
        """
        return self.mu_mpo.forward(x, to_tensor=to_tensor)
    
    def forward_sigma(self, to_tensor=False):
        """
        Forward pass through Σ-MPO.
        
        Σ-MPO represents the correlation/variation structure.
        
        Args:
            to_tensor: If True, return tensor instead of TensorNode
            
        Returns:
            Output from Σ-MPO (correlation structure)
        """
        return self.sigma_mpo.forward(None, to_tensor=to_tensor)
    
    def get_mu_jacobian(self, node):
        """
        Get Jacobian for a μ-MPO node.
        
        Args:
            node: μ-MPO node
            
        Returns:
            Jacobian (network contracted without this node)
        """
        return self.mu_mpo.get_jacobian(node)
    
    def get_sigma_jacobian(self, node):
        """
        Get Jacobian for a Σ-MPO node.
        
        Args:
            node: Σ-MPO node
            
        Returns:
            Jacobian (network contracted without this node)
        """
        return self.sigma_mpo.compute_jacobian_stack(node)
    
    def trim(self, thresholds):
        """
        Trim both μ-MPO and Σ-MPO based on μ expectations.
        
        When trimming μ-MPO dimension 'r1' to keep indices [0, 2],
        Σ-MPO dimensions 'r1o' and 'r1i' are both trimmed to keep [0, 2].
        
        Args:
            thresholds: Dict mapping dim_label -> threshold value
        """
        # First, determine which indices to keep from μ-MPO
        keep_indices = {}
        
        for label, dist in self.mu_mpo.distributions.items():
            threshold = thresholds.get(label, float('-inf'))
            expectations = dist['expectation']
            
            mask = expectations >= threshold
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                raise ValueError(
                    f"Trimming dimension '{label}' with threshold {threshold} "
                    f"would remove all indices."
                )
            
            keep_indices[label] = indices
        
        # Trim μ-MPO
        self.mu_mpo.trim(thresholds)
        
        # Trim Σ-MPO (both 'o' and 'i' versions)
        sigma_keep_indices = {}
        for label, indices in keep_indices.items():
            sigma_keep_indices[f"{label}o"] = indices
            sigma_keep_indices[f"{label}i"] = indices
        
        for node in self.sigma_nodes:
            self._trim_sigma_node(node, sigma_keep_indices)
        
        # Reset stacks
        self.sigma_mpo.reset_stacks()
        
        return self
    
    def _trim_sigma_node(self, node, keep_indices):
        """Trim a Σ-MPO node based on keep_indices."""
        new_tensor = node.tensor
        
        for dim_idx, label in enumerate(node.dim_labels):
            if label in keep_indices:
                indices = keep_indices[label]
                new_tensor = torch.index_select(new_tensor, dim_idx, indices)
        
        node.tensor = new_tensor
    
    def to(self, device=None, dtype=None):
        """Move all structures to device/dtype."""
        self.mu_mpo.to(device=device, dtype=dtype)
        self.sigma_mpo.to(device=device, dtype=dtype)
        if device is not None or dtype is not None:
            self.tau_alpha = self.tau_alpha.to(device=device, dtype=dtype)
            self.tau_beta = self.tau_beta.to(device=device, dtype=dtype)
            self.tau_distribution = GammaDistribution(
                concentration=self.tau_alpha,
                rate=self.tau_beta
            )
        return self
    
    def summary(self):
        """Print summary of the Bayesian MPO structure."""
        print("=" * 70)
        print("Bayesian MPO Summary")
        print("=" * 70)
        
        print(f"\nμ-MPO:")
        print(f"  Number of nodes: {len(self.mu_nodes)}")
        for i, node in enumerate(self.mu_nodes):
            print(f"  Node {i}: {node.shape} with labels {node.dim_labels}")
        
        print(f"\nΣ-MPO (doubled structure):")
        print(f"  Number of nodes: {len(self.sigma_nodes)}")
        for i, node in enumerate(self.sigma_nodes):
            print(f"  Node {i}: {node.shape} with labels {node.dim_labels}")
        
        print(f"\nτ distribution: Gamma(α={self.tau_alpha.item():.2f}, β={self.tau_beta.item():.2f})")
        print(f"  E[τ] = {self.get_tau_mean():.4f}")
        print(f"  H[τ] = {self.get_tau_entropy():.4f}")
        
        print(f"\nPrior distributions (μ-MPO):")
        for label, dist in self.mu_mpo.distributions.items():
            print(f"  {label}: {dist['type']} ({dist['variable']}), size={len(dist['expectation'])}")
