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
from tensor.probability_distributions import (
    GammaDistribution, 
    MultivariateGaussianDistribution,
    ProductDistribution
)
from typing import Dict, List, Optional, Tuple, Union, Set, Any


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
    
    def __init__(
        self, 
        mu_nodes: List[TensorNode], 
        input_nodes: Optional[List[TensorNode]] = None, 
        rank_labels: Optional[Union[List[str], set]] = None, 
        tau_alpha: Optional[torch.Tensor] = None, 
        tau_beta: Optional[torch.Tensor] = None
    ) -> None:
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
    
    def _create_sigma_nodes(self) -> List[TensorNode]:
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
        
        # Initialize Σ-nodes to ensure positive definite covariance
        self._initialize_sigma_nodes(sigma_nodes)
        
        return sigma_nodes
    
    def _initialize_sigma_nodes(self, sigma_nodes: List[TensorNode]) -> None:
        """
        Initialize Σ-MPO nodes to ensure covariance is positive definite.
        
        For each block, we want Cov = Σ - μ ⊗ μ^T to be positive definite.
        A simple initialization is: Σ = μ ⊗ μ^T + σ²I
        where σ² is a small variance parameter.
        
        This corresponds to setting the Σ-node such that when reshaped:
        E[W ⊗ W^T] = μ ⊗ μ^T + σ²I
        
        Args:
            sigma_nodes: List of Σ-MPO nodes to initialize
        """
        for mu_node, sigma_node in zip(self.mu_nodes, sigma_nodes):
            # Get μ block (flattened)
            mu_flat = mu_node.tensor.flatten()
            d = mu_flat.numel()
            
            # Compute μ ⊗ μ^T + σ²I
            # Using σ² = 1.0 as default initial variance
            initial_variance = 1.0
            mu_outer = torch.outer(mu_flat, mu_flat)
            sigma_matrix = mu_outer + initial_variance * torch.eye(d, dtype=mu_flat.dtype, device=mu_flat.device)
            
            # Reshape sigma_matrix (d, d) back to Σ-node shape
            # Σ-node has shape (d1, d1, d2, d2, ...) where d = d1 * d2 * ...
            shape_half = mu_node.shape
            n_dims = len(shape_half)
            
            # First reshape to separate dimensions
            # (d, d) -> (d1, d2, ..., d1, d2, ...)
            expanded_shape = list(shape_half) + list(shape_half)
            sigma_expanded = sigma_matrix.reshape(*expanded_shape)
            
            # Permute to interleave outer and inner: (d1, d2, ..., d1, d2, ...) -> (d1, d1, d2, d2, ...)
            # Create permutation that interleaves
            perm = []
            for i in range(n_dims):
                perm.append(i)           # outer dimension i
                perm.append(i + n_dims)  # inner dimension i
            
            sigma_permuted = sigma_expanded.permute(*perm)
            
            # Set the Σ-node tensor
            sigma_node.tensor = sigma_permuted
    
    def get_mu_distributions(self, label: str) -> Optional[List[GammaDistribution]]:
        """Get Gamma distributions for a μ-MPO dimension."""
        return self.mu_mpo.get_gamma_distributions(label)
    
    def get_mu_expectations(self, label: str) -> Optional[torch.Tensor]:
        """Get expectation values E[ω] or E[φ] for μ-MPO dimension."""
        return self.mu_mpo.get_expectations(label)
    
    def update_mu_params(
        self, 
        label: str, 
        param1: Optional[torch.Tensor] = None, 
        param2: Optional[torch.Tensor] = None
    ) -> None:
        """Update μ-MPO distribution parameters."""
        self.mu_mpo.update_distribution_params(label, param1, param2)
    
    def update_tau(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        beta: Optional[torch.Tensor] = None
    ) -> None:
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
    
    def get_tau_mean(self) -> torch.Tensor:
        """Get mean of τ distribution: E[τ] = α/β."""
        return self.tau_distribution.mean()
    
    def get_tau_entropy(self) -> torch.Tensor:
        """Get entropy of τ distribution."""
        return self.tau_distribution.entropy()
    
    def forward_mu(self, x: torch.Tensor, to_tensor: bool = False) -> Union[TensorNode, torch.Tensor]:
        """
        Forward pass through μ-MPO.
        
        Args:
            x: Input data
            to_tensor: If True, return tensor instead of TensorNode
            
        Returns:
            Output from μ-MPO
        """
        return self.mu_mpo.forward(x, to_tensor=to_tensor)
    
    def forward_sigma(self, to_tensor: bool = False) -> Union[TensorNode, torch.Tensor]:
        """
        Forward pass through Σ-MPO.
        
        Σ-MPO represents the correlation/variation structure.
        
        Args:
            to_tensor: If True, return tensor instead of TensorNode
            
        Returns:
            Output from Σ-MPO (correlation structure)
        """
        return self.sigma_mpo.forward(None, to_tensor=to_tensor)
    
    def get_mu_jacobian(self, node: TensorNode):
        """
        Get Jacobian for a μ-MPO node.
        
        Args:
            node: μ-MPO node
            
        Returns:
            Jacobian (network contracted without this node)
        """
        return self.mu_mpo.get_jacobian(node)
    
    def get_sigma_jacobian(self, node: TensorNode):
        """
        Get Jacobian for a Σ-MPO node.
        
        Args:
            node: Σ-MPO node
            
        Returns:
            Jacobian (network contracted without this node)
        """
        return self.sigma_mpo.compute_jacobian_stack(node)
    
    def trim(self, thresholds: Dict[str, float]) -> 'BayesianMPO':
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
    
    def _trim_sigma_node(self, node: TensorNode, keep_indices: Dict[str, torch.Tensor]) -> None:
        """Trim a Σ-MPO node based on keep_indices."""
        new_tensor = node.tensor
        
        for dim_idx, label in enumerate(node.dim_labels):
            if label in keep_indices:
                indices = keep_indices[label]
                new_tensor = torch.index_select(new_tensor, dim_idx, indices)
        
        node.tensor = new_tensor
    
    def to(
        self, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ) -> 'BayesianMPO':
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
    
    def get_block_q_distribution(self, block_idx: int) -> MultivariateGaussianDistribution:
        """
        Get the q-distribution for a specific block as a Multivariate Normal.
        
        For block i, q(W_i) = N(μ_i, Σ_i - μ_i ⊗ μ_i^T)
        where:
        - μ_i = E_q[W_i] is the μ-MPO block (flattened)
        - Σ_i = E_q[W_i ⊗ W_i^T] is related to the Σ-MPO block (flattened)
        - Covariance = Σ_i - μ_i ⊗ μ_i^T
        
        Args:
            block_idx: Index of the block (0 to N-1)
            
        Returns:
            MultivariateGaussianDistribution for this block
        """
        mu_node = self.mu_nodes[block_idx]
        sigma_node = self.sigma_nodes[block_idx]
        
        # Flatten μ block: E_q[W_i]
        mu_flat = mu_node.tensor.flatten()
        d = mu_flat.numel()
        
        # Flatten Σ block to get E_q[W_i ⊗ W_i^T]
        # Σ-node has shape (d1, d1, d2, d2, ...) -> reshape to (d, d)
        sigma_tensor = sigma_node.tensor
        
        # Reshape: group 'o' and 'i' dimensions
        # For each original dimension of size n, we have (n_o, n_i)
        # We want to flatten to (d, d) where d = product of all n's
        shape_half = mu_node.shape
        
        # Permute to group outer and inner: [d1_o, d2_o, ..., d1_i, d2_i, ...]
        n_dims = len(shape_half)
        outer_indices = list(range(0, 2*n_dims, 2))  # [0, 2, 4, ...]
        inner_indices = list(range(1, 2*n_dims, 2))  # [1, 3, 5, ...]
        perm = outer_indices + inner_indices
        
        sigma_permuted = sigma_tensor.permute(*perm)
        
        # Reshape to (d, d)
        sigma_matrix = sigma_permuted.reshape(d, d)
        
        # Compute covariance: Cov = E[W ⊗ W^T] - E[W] ⊗ E[W]^T
        mu_outer = torch.outer(mu_flat, mu_flat)
        covariance = sigma_matrix - mu_outer
        
        # Make symmetric to handle numerical errors (floating point precision)
        covariance = 0.5 * (covariance + covariance.T)
        
        return MultivariateGaussianDistribution(
            loc=mu_flat,
            covariance_matrix=covariance
        )
    
    def get_all_block_q_distributions(self) -> List[MultivariateGaussianDistribution]:
        """
        Get q-distributions for all blocks.
        
        Returns:
            List of MultivariateGaussianDistribution, one for each block
        """
        return [self.get_block_q_distribution(i) for i in range(len(self.mu_nodes))]
    
    def get_mode_q_distributions(self) -> Dict[str, List[GammaDistribution]]:
        """
        Get all Gamma q-distributions for each mode (dimension).
        
        For each mode (dimension label), we have a list of Gamma distributions,
        one for each index in that dimension.
        
        Returns:
            Dict mapping dimension label -> list of GammaDistribution objects
        """
        mode_distributions = {}
        for label in self.mu_mpo.distributions.keys():
            gammas = self.mu_mpo.get_gamma_distributions(label)
            if gammas is not None:
                mode_distributions[label] = gammas
        return mode_distributions
    
    def get_full_q_distribution(self) -> ProductDistribution:
        """
        Get the full q-distribution as a product of all components.
        
        q(θ) = q(τ) × ∏_i q(W_i) × ∏_modes ∏_indices q(mode_param)
        
        where:
        - q(τ) ~ Gamma(α, β)
        - q(W_i) ~ N(μ_i, Σ_i - μ_i ⊗ μ_i^T) for each block i
        - q(mode_param) ~ Gamma(c, e) or Gamma(f, g) for each mode index
        
        Returns:
            ProductDistribution containing all component distributions
        """
        all_distributions = []
        
        # 1. Add τ distribution
        all_distributions.append(self.tau_distribution)
        
        # 2. Add all block distributions (Multivariate Normals)
        block_dists = self.get_all_block_q_distributions()
        all_distributions.extend(block_dists)
        
        # 3. Add all mode distributions (Gammas)
        mode_dists = self.get_mode_q_distributions()
        for label, gamma_list in mode_dists.items():
            all_distributions.extend(gamma_list)
        
        return ProductDistribution(all_distributions)
    
    def sample_from_q(self, n_samples: int = 1) -> Dict[str, Any]:
        """
        Sample from the q-distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary containing:
            - 'tau': Samples from τ distribution
            - 'blocks': List of samples from each block distribution
            - 'modes': Dict of samples from each mode distribution
        """
        samples = {}
        
        # Sample τ
        samples['tau'] = self.tau_distribution.forward().sample((n_samples,))
        
        # Sample blocks
        block_dists = self.get_all_block_q_distributions()
        samples['blocks'] = [
            dist.forward().sample((n_samples,)) for dist in block_dists
        ]
        
        # Sample modes
        mode_dists = self.get_mode_q_distributions()
        samples['modes'] = {}
        for label, gamma_list in mode_dists.items():
            samples['modes'][label] = [
                gamma.forward().sample((n_samples,)) for gamma in gamma_list
            ]
        
        return samples
    
    def log_q(self, theta: Dict[str, Any]) -> torch.Tensor:
        """
        Compute log q(θ) for given parameter values.
        
        Args:
            theta: Dictionary containing:
                - 'tau': τ values
                - 'blocks': List of block parameter values (flattened)
                - 'modes': Dict of mode parameter values
                
        Returns:
            log q(θ) = log q(τ) + Σ log q(W_i) + Σ log q(mode_params)
        """
        log_prob = torch.tensor(0.0, dtype=self.tau_alpha.dtype)
        
        # Add log q(τ)
        log_prob = log_prob + self.tau_distribution.forward().log_prob(theta['tau'])
        
        # Add log q(W_i) for each block
        block_dists = self.get_all_block_q_distributions()
        for i, (dist, block_val) in enumerate(zip(block_dists, theta['blocks'])):
            log_prob = log_prob + dist.forward().log_prob(block_val)
        
        # Add log q(mode_params)
        mode_dists = self.get_mode_q_distributions()
        for label, gamma_list in mode_dists.items():
            mode_values = theta['modes'][label]
            for gamma, val in zip(gamma_list, mode_values):
                log_prob = log_prob + gamma.forward().log_prob(val)
        
        return log_prob
    
    def summary(self) -> None:
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
