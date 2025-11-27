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
        
        # Initialize prior hyperparameters (uninformative by default)
        self._initialize_prior_hyperparameters()
    
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

    def _get_sigma_to_mu_permutation(self, block_idx: int) -> Tuple[List[int], List[int]]:
        """
        Get permutation to reshape Σ-block to match μ-block structure using labels.
        
        Σ-node has labels like ['r1o', 'r1i', 'c2o', 'c2i', 'p2o', 'p2i', 'r2o', 'r2i']
        μ-node has labels like ['r1', 'c2', 'p2', 'r2']
        
        This function finds which Σ dimensions correspond to outer and inner for each μ dimension.
        
        Args:
            block_idx: Index of the block
            
        Returns:
            Tuple of (outer_indices, inner_indices) where:
            - outer_indices[i] is the Σ dimension for μ dimension i (outer)
            - inner_indices[i] is the Σ dimension for μ dimension i (inner)
            
        Example:
            μ labels: ['r1', 'c2', 'p2', 'r2']
            Σ labels: ['r1o', 'r1i', 'c2o', 'c2i', 'p2o', 'p2i', 'r2o', 'r2i']
            Returns: ([0, 2, 4, 6], [1, 3, 5, 7])
        """
        mu_node = self.mu_nodes[block_idx]
        sigma_node = self.sigma_nodes[block_idx]
        
        outer_indices = []
        inner_indices = []
        
        for mu_label in mu_node.dim_labels:
            # Find corresponding 'o' and 'i' labels in sigma
            label_o = f"{mu_label}o"
            label_i = f"{mu_label}i"
            
            # Get their indices in sigma
            try:
                idx_o = sigma_node.dim_labels.index(label_o)
                idx_i = sigma_node.dim_labels.index(label_i)
            except ValueError as e:
                raise ValueError(f"Cannot find labels {label_o} or {label_i} in Σ-node labels {sigma_node.dim_labels}")
            
            outer_indices.append(idx_o)
            inner_indices.append(idx_i)
        
        return outer_indices, inner_indices
    
    def _sigma_to_matrix(self, block_idx: int) -> torch.Tensor:
        """
        Convert Σ-block tensor to (d, d) matrix using label-based permutation.
        
        Uses labels to correctly identify outer and inner dimensions, then
        permutes to group them and reshapes to matrix form.
        
        Args:
            block_idx: Index of the block
            
        Returns:
            Matrix of shape (d, d) where d is the flattened size of μ-block
        """
        mu_node = self.mu_nodes[block_idx]
        sigma_node = self.sigma_nodes[block_idx]
        
        # Get permutation using labels
        outer_indices, inner_indices = self._get_sigma_to_mu_permutation(block_idx)
        
        # Permute: [outer dims, then inner dims]
        perm = outer_indices + inner_indices
        sigma_permuted = sigma_node.tensor.permute(*perm)
        
        # Reshape to (d, d)
        d = mu_node.tensor.numel()
        sigma_matrix = sigma_permuted.reshape(d, d)
        
        return sigma_matrix

    def _matrix_to_sigma(self, block_idx: int, matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert (d, d) matrix back to Σ-block tensor shape.
        
        This is the inverse operation of _sigma_to_matrix.
        
        Args:
            block_idx: Index of the block
            matrix: Matrix of shape (d, d)
            
        Returns:
            Tensor with Σ-block shape (d1_o, d1_i, d2_o, d2_i, ...)
        """
        mu_node = self.mu_nodes[block_idx]
        sigma_node = self.sigma_nodes[block_idx]
        
        mu_shape = mu_node.shape
        d = mu_node.tensor.numel()
        n_dims = len(mu_shape)
        
        # Reshape matrix (d, d) to grouped form
        # First: (d, d) -> (d1, d2, ..., d1, d2, ...)
        expanded_shape = list(mu_shape) + list(mu_shape)
        matrix_expanded = matrix.reshape(*expanded_shape)
        
        # Get inverse permutation
        # Forward was: outer_indices + inner_indices
        # Inverse: interleave them back
        outer_indices, inner_indices = self._get_sigma_to_mu_permutation(block_idx)
        
        # Create inverse permutation
        # We went from interleaved [o, i, o, i, ...] to grouped [o, o, ..., i, i, ...]
        # Now go back: from grouped to interleaved
        inv_perm = [0] * (2 * n_dims)
        for i, (o_idx, i_idx) in enumerate(zip(outer_indices, inner_indices)):
            inv_perm[o_idx] = i  # Outer goes to position i
            inv_perm[i_idx] = i + n_dims  # Inner goes to position i + n_dims
        
        sigma_tensor = matrix_expanded.permute(*inv_perm)
        
        return sigma_tensor
    
    def _initialize_prior_hyperparameters(self) -> None:
        """
        Initialize prior hyperparameters p(θ).
        
        The prior should be set BEFORE variational parameters, as priors inform initialization.
        Default prior hyperparameters (uninformative/weakly informative):
        - p(τ) ~ Gamma(α₀=1, β₀=1)  - uninformative
        - p(Wᵢ) ~ N(0, σ₀²I) where σ₀²=10 - weakly informative, large variance
        - p(bond params) ~ Gamma(1, 1) - uninformative, for each index in each bond
        
        These should be set via set_prior_hyperparameters() before running inference.
        """
        # Prior for τ: uninformative Gamma(1, 1)
        self.prior_tau_alpha = torch.tensor(1.0, dtype=self.tau_alpha.dtype, device=self.tau_alpha.device)
        self.prior_tau_beta = torch.tensor(1.0, dtype=self.tau_beta.dtype, device=self.tau_beta.device)
        
        # Prior for bond parameters: uninformative Gamma(1, 1) for all bonds
        # Store as dict: {label: {'concentration0': tensor, 'rate0': tensor}}
        # Each bond has N Gamma distributions (one per index)
        self.prior_bond_params = {}
        for label, dist_info in self.mu_mpo.distributions.items():
            size = len(dist_info['expectation'])
            self.prior_bond_params[label] = {
                'concentration0': torch.ones(size, dtype=dist_info['concentration'].dtype, 
                                            device=dist_info['concentration'].device),
                'rate0': torch.ones(size, dtype=dist_info['rate'].dtype, 
                                   device=dist_info['rate'].device)
            }
        
        # Prior for blocks: N(0, σ₀²I) with σ₀²=10 (weakly informative)
        # TODO: Since the dimension of the Sigma block diverge very rapidly, and the sigma prior is just diagonal, 
        # it is not needed to expand it and store. we can simply store the diagonal values and If isomorphic, 
        # only the value of the variance without expliciting the identity. Propagate the usage of this logic 
        # for where it gets used. Also variance should be made as a choosable parameter.
        self.prior_block_sigma0 = []
        for mu_node in self.mu_nodes:
            d = mu_node.tensor.numel()
            # Prior covariance: 10 * I (large variance, weakly informative)
            sigma0 = 10.0 * torch.eye(d, dtype=mu_node.tensor.dtype, device=mu_node.tensor.device)
            self.prior_block_sigma0.append(sigma0)
    
    def set_prior_hyperparameters(
        self,
        tau_alpha0: Optional[torch.Tensor] = None,
        tau_beta0: Optional[torch.Tensor] = None,
        bond_params0: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        block_sigma0: Optional[List[torch.Tensor]] = None
    ) -> None:
        """
        Set prior hyperparameters for p(θ).
        
        Args:
            tau_alpha0: Prior α₀ for p(τ) ~ Gamma(α₀, β₀)
            tau_beta0: Prior β₀ for p(τ) ~ Gamma(α₀, β₀)
            bond_params0: Dict of prior parameters for bonds, e.g.:
                         {'r1': {'concentration0': tensor, 'rate0': tensor}}
                         Each tensor has length N (one param per index)
            block_sigma0: List of prior covariances Σ₀ for p(Wᵢ) ~ N(0, Σ₀)
        """
        if tau_alpha0 is not None:
            self.prior_tau_alpha = tau_alpha0
        if tau_beta0 is not None:
            self.prior_tau_beta = tau_beta0
        if bond_params0 is not None:
            self.prior_bond_params.update(bond_params0)
        if block_sigma0 is not None:
            self.prior_block_sigma0 = block_sigma0
    
    def get_mu_distributions(self, label: str) -> Optional[List[GammaDistribution]]:
        """Get Gamma distributions for a μ-MPO dimension."""
        return self.mu_mpo.get_gamma_distributions(label)
    
    def get_mu_expectations(self, label: str) -> Optional[torch.Tensor]:
        """Get expectation values E[ω] or E[φ] for μ-MPO dimension."""
        return self.mu_mpo.get_expectations(label)
    
    def update_mu_params(
        self, 
        label: str, 
        concentration: Optional[torch.Tensor] = None, 
        rate: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update μ-MPO distribution parameters for a bond.
        
        Args:
            label: Bond label
            concentration: Concentration parameters (α)
            rate: Rate parameters (β)
        """
        self.mu_mpo.update_distribution_params(label, concentration, rate)

    def get_nodes_for_bond(self, label: str) -> Dict[str, List[int]]:
        """
        Get node indices associated with a given bond for both μ-MPO and Σ-MPO.
        
        Args:
            label: Bond label (e.g., 'r1', 'f1')
            
        Returns:
            Dictionary with keys:
            - 'mu_nodes': List of μ-MPO node indices sharing this bond
            - 'sigma_nodes': List of Σ-MPO node indices sharing this bond (same as mu_nodes)
        """
        mu_node_indices = self.mu_mpo.get_nodes_for_bond(label)
        
        # Σ-MPO nodes correspond 1-to-1 with μ-MPO nodes
        # The same node indices apply to both
        sigma_node_indices = mu_node_indices.copy()
        
        return {
            'mu_nodes': mu_node_indices,
            'sigma_nodes': sigma_node_indices
        }

    def compute_theta_tensor(self, block_idx: int, exclude_labels: Optional[List[str]] = None) -> torch.Tensor:
        """
        Compute the Theta tensor for a μ-MPO block from Gamma expectations.
        
        Theta_{ijk...} = E_q[bond1]_i × E_q[bond2]_j × E_q[bond3]_k × ...
        
        Args:
            block_idx: Index of the μ-MPO block (0 to N-1)
            exclude_labels: List of bond labels to exclude (set to dimension 1)
            
        Returns:
            Theta tensor matching the block shape
            
        Example:
            # Block 1 has bonds ['r1', 'c2', 'p2', 'r2'] with shape (4, 1, 5, 4)
            theta = bmpo.compute_theta_tensor(1)
            # Shape: (4, 1, 5, 4) from outer product of [E[r1], 1, E[p2], E[r2]]
            
            theta_no_p = bmpo.compute_theta_tensor(1, exclude_labels=['p2'])
            # Shape: (4, 1, 1, 4), excludes p2 contribution
        """
        return self.mu_mpo.compute_theta_tensor(block_idx, exclude_labels)

    def partial_trace_update(self, block_idx: int, focus_label: str) -> torch.Tensor:
        """
        Compute partial trace for variational update of a specific bond.
        
        This computes: Σ_{other indices} [diag(Σ) + μ²] × Θ_{without focus_label}
        
        where the sum is over all dimensions except focus_label, and the multiplication
        aligns matching labels.
        
        Algorithm:
        1. Get diagonal of Σ-block: diag(Σ_{ijk...} with paired (io, ii))
        2. Add μ² element-wise: v = diag(Σ) + μ²
        3. Get Θ excluding focus_label
        4. Multiply: v × Θ (element-wise, matching dimensions)
        5. Sum over all dimensions except focus_label
        
        Args:
            block_idx: Index of the block
            focus_label: The bond label to keep (not sum over)
            
        Returns:
            Vector of length equal to the dimension of focus_label
            
        Example:
            Block has shape (4, 1, 5, 4) with labels ['r1', 'c2', 'p2', 'r2']
            partial_trace_update(1, 'p2') returns vector of length 5
            
            This computes: Σ_{i,j,k} [diag(Σ) + μ²]_{ijk𝓁} × Θ_{ijk𝓁 without p2}
            where the sum is over r1(i), c2(j), r2(k), keeping only p2(𝓁)
        """
        mu_node = self.mu_nodes[block_idx]
        
        # Check that focus_label is in this block
        if focus_label not in mu_node.dim_labels:
            raise ValueError(f"Label '{focus_label}' not found in block {block_idx}. "
                           f"Available labels: {mu_node.dim_labels}")
        
        # Step 1: Extract diagonal of Σ-block using label-based method
        sigma_matrix = self._sigma_to_matrix(block_idx)  # Shape: (d, d)
        diag_sigma_flat = torch.diagonal(sigma_matrix)  # Shape: (d,)
        
        # Reshape back to block shape
        mu_shape = mu_node.shape
        diag_sigma = diag_sigma_flat.reshape(mu_shape)
        
        # Step 2: Add μ² element-wise
        mu_tensor = mu_node.tensor
        v = diag_sigma + mu_tensor ** 2  # Shape: same as mu_shape
        
        # Step 3: Get Θ excluding focus_label
        theta = self.compute_theta_tensor(block_idx, exclude_labels=[focus_label])
        # Theta has shape where focus_label dimension is 1
        
        # Step 4: Multiply element-wise (broadcasting handles the dimension of size 1)
        result = v * theta  # Shape: same as mu_shape
        
        # Step 5: Sum over all dimensions except focus_label
        # Find which dimension index corresponds to focus_label
        focus_dim_idx = mu_node.dim_labels.index(focus_label)
        
        # Sum over all dimensions except focus_dim_idx
        dims_to_sum = [i for i in range(len(mu_shape)) if i != focus_dim_idx]
        output = torch.sum(result, dim=dims_to_sum)
        
        return output

    def update_bond_variational(self, label: str) -> None:
        """
        Variational update for a bond's Gamma distribution parameters.
        
        Updates q(bond) ~ Gamma(concentration_q, rate_q) based on the current
        μ and Σ values.
        
        Update formulas:
        - concentration_q = concentration_p + N_b × dim(bond)
        - rate_q = rate_p - Σ_{blocks with bond} partial_trace_update(block, label)
        
        where:
        - N_b is the number of learnable blocks associated with the bond (1 or 2)
        - dim(bond) is the dimension size of the bond
        - concentration_p, rate_p are prior hyperparameters
        - partial_trace_update contracts over all dimensions except the bond
        
        Args:
            label: Bond label to update (e.g., 'r1', 'p2')
            
        Example:
            # Bond 'r1' connects blocks [0, 1], has dimension 4
            bmpo.update_bond_variational('r1')
            
            # Updates:
            # concentration_q = concentration_p + 2 × 4 = concentration_p + 8
            # rate_q = rate_p - partial_trace(block=0, 'r1') 
            #                 - partial_trace(block=1, 'r1')
        """
        # Check that the bond exists and has Gamma distribution
        if label not in self.mu_mpo.distributions:
            raise ValueError(f"Bond '{label}' not found in distributions. "
                           f"Available bonds: {list(self.mu_mpo.distributions.keys())}")
        
        # Get bond information
        bond_dist = self.mu_mpo.distributions[label]
        bond_size = len(bond_dist['concentration'])
        
        # Get blocks associated with this bond
        nodes_info = self.get_nodes_for_bond(label)
        block_indices = nodes_info['mu_nodes']
        N_b = len(block_indices)  # Number of blocks (1 or 2)
        
        # Get prior parameters
        if label not in self.prior_bond_params:
            raise ValueError(f"No prior parameters found for bond '{label}'")
        
        prior_params = self.prior_bond_params[label]
        concentration_p = prior_params['concentration0']  # Shape: (bond_size,)
        rate_p = prior_params['rate0']  # Shape: (bond_size,)
        
        # Update formula 1: concentration_q = concentration_p + N_b × dim(bond)
        concentration_q = concentration_p + N_b * bond_size
        
        # Update formula 2: rate_q = rate_p - Σ_{blocks} partial_trace_update
        rate_q = rate_p.clone()
        
        for block_idx in block_indices:
            # Compute partial trace for this block with focus on the bond
            partial_trace = self.partial_trace_update(block_idx, label)
            # Shape: (bond_size,)
            
            # ADD to rate_q
            rate_q = rate_q + partial_trace
        
        # Update the bond parameters
        self.update_mu_params(label, concentration=concentration_q, rate=rate_q)

    def _compute_mu_jacobian_outer(
        self,
        node: TensorNode,
        network: 'TensorNetwork',
        output_shape: tuple
    ) -> torch.Tensor:
        """
        Compute Σₙ J_μ(xₙ) ⊗ J_μ(xₙ) - outer product of Jacobian with itself.
        
        This is J^T @ J summed over samples, without hessian weighting.
        """
        from tensor.utils import EinsumLabeler
        
        # Get Jacobian
        J = network.compute_jacobian_stack(node).copy()
        
        # Expand to include output labels
        J = J.expand_labels(network.output_labels, output_shape)
        
        broadcast_dims = tuple(d for d in network.output_labels if d not in node.dim_labels)
        J = J.permute_first(*broadcast_dims)
        
        # Build einsum for J^T @ J
        dim_labels = EinsumLabeler()
        
        J_ein1 = ''.join([dim_labels[d] for d in J.dim_labels])
        J_ein2 = ''.join([dim_labels['_' + d] if d != network.sample_dim else dim_labels[d] for d in J.dim_labels])
        
        # Output: node dimensions twice
        J_out1 = []
        J_out2 = []
        dim_order = []
        for d in J.dim_labels:
            if d not in broadcast_dims:
                J_out1.append(dim_labels[d])
                J_out2.append(dim_labels['_' + d])
                dim_order.append(d)
        
        out1 = ''.join([J_out1[dim_order.index(d)] for d in node.dim_labels])
        out2 = ''.join([J_out2[dim_order.index(d)] for d in node.dim_labels])
        
        # Einsum: sum over samples, outer product over node dims
        einsum_str = f'{J_ein1},{J_ein2}->{out1}{out2}'
        
        A = torch.einsum(einsum_str, J.tensor.conj(), J.tensor)
        
        return A
    
    def _compute_forward_without_node(
        self,
        node: TensorNode,
        network: 'TensorNetwork'
    ) -> TensorNode:
        """
        Compute forward pass through network without contracting a specific node.
        
        This gives us the Jacobian: when contracted with the node, produces the output.
        Uses left and right stacks for efficient computation.
        
        Args:
            node: Node to exclude from contraction
            network: TensorNetwork (mu_mpo or sigma_mpo)
            
        Returns:
            Jacobian as TensorNode
        """
        # Get left and right stacks
        if network.left_stacks is None or network.right_stacks is None:
            network.recompute_all_stacks()
        
        left_stack, right_stack = network.get_stacks(node)
        
        # Get column nodes (vertical connections)
        column_nodes = network.get_column_nodes(node)
        
        # Start with left stack (or first column node if no left stack)
        node_iter = iter(column_nodes)
        contracted = next(node_iter) if left_stack is None else left_stack
        
        # Contract with remaining column nodes
        for vnode in node_iter:
            contracted = contracted.contract_with(vnode, vnode.get_connecting_labels(contracted))
        
        # Contract with right stack
        if right_stack is not None:
            contracted = contracted.contract_with(right_stack, right_stack.get_connecting_labels(contracted))
        
        return contracted
    
    def _compute_jacobian_y_contraction(
        self,
        node: TensorNode,
        y: torch.Tensor,
        network: 'TensorNetwork'
    ) -> torch.Tensor:
        """
        Compute Σₙ yₙ · J(xₙ) where J is the Jacobian (forward without node).
        
        This contracts J with y over the sample dimension, leaving node dimensions.
        """
        from tensor.utils import EinsumLabeler
        
        # Get Jacobian: network output without this node
        J = self._compute_forward_without_node(node, network)
        
        # J has sample dimension 's' and dimensions from other parts of network
        # We need to contract J with y over samples, leaving only node-shaped output
        
        # Build einsum string
        labeler = EinsumLabeler()
        
        # J einsum string
        J_ein = ''.join([labeler[d] for d in J.dim_labels])
        
        # y einsum string (sample dimension)
        y_ein = labeler['s']  # y has shape (S,) or (S, 1)
        
        # Output: only dimensions that are in the node
        out_dims = []
        for d in J.dim_labels:
            if d != 's' and d in node.dim_labels:
                out_dims.append(d)
        out_ein = ''.join([labeler[d] for d in out_dims])
        
        # Contract: sum over samples and any non-node dimensions
        einsum_str = f"{J_ein},{y_ein}->{out_ein}"
        
        result = torch.einsum(einsum_str, J.tensor, y.squeeze(-1) if y.dim() > 1 else y)
        
        return result
    
    def _compute_jacobian_outer_product(
        self,
        node: TensorNode,
        network: 'TensorNetwork'
    ) -> torch.Tensor:
        """
        Compute Σₙ J(xₙ) ⊗ J(xₙ) where J is the Jacobian.
        
        This computes J^T @ J contracted over samples, leaving (node_dims, node_dims).
        """
        from tensor.utils import EinsumLabeler
        
        # Get Jacobian
        J = self._compute_forward_without_node(node, network)
        
        # Build einsum for outer product
        labeler = EinsumLabeler()
        
        # J einsum strings (same J used twice with different labels for outer product)
        J_ein1 = ''.join([labeler[d] for d in J.dim_labels])
        J_ein2 = ''.join([labeler['_' + d] if d != 's' else labeler[d] for d in J.dim_labels])
        
        # Output: node dimensions twice (outer product)
        out_dims1 = [d for d in J.dim_labels if d != 's' and d in node.dim_labels]
        out_dims2 = ['_' + d for d in out_dims1]
        out_ein = ''.join([labeler[d] for d in out_dims1]) + ''.join([labeler[d] for d in out_dims2])
        
        # Contract
        einsum_str = f"{J_ein1},{J_ein2}->{out_ein}"
        
        result = torch.einsum(einsum_str, J.tensor, J.tensor)
        
        return result
    
    def update_block_variational(
        self, 
        block_idx: int, 
        X: torch.Tensor, 
        y: torch.Tensor
    ) -> None:
        """
        Variational update for a block's parameters (μ and Σ).
        
        Updates the natural parameters for block χⁱ based on data (X, y).
        
        Update equations:
        - μ^χⁱ = -E[τ] Σ^χⁱ Σₙ yₙ · J_μ(xₙ)
        - Σ^χⁱ⁻¹ = -E[τ] Σₙ [J_Σ(xₙ) + J_μ(xₙ) ⊗ J_μ(xₙ)] - diag(Θ)
        
        where:
        - J_μ(xₙ) = ∂/∂χⁱ μ(xₙ) is the μ-Jacobian (network with block i removed)
        - J_Σ(xₙ) = ∂/∂χⁱ Σ(xₙ⊗xₙ) is the Σ-Jacobian
        - Θ = theta tensor (outer product of Gamma expectations)
        - diag(Θ) = diagonal matrix from Θ
        
        Args:
            block_idx: Index of the block to update (0 to N-1)
            X: Input data, shape (S, feature_dims) where S is number of samples
            y: Target data, shape (S,) for scalar outputs
            
        Example:
            # Update block 1 with training data
            bmpo.update_block_variational(1, X_train, y_train)
        """
        mu_node = self.mu_nodes[block_idx]
        sigma_node = self.sigma_nodes[block_idx]
        
        # Get E[τ]
        E_tau = self.get_tau_mean()
        
        # Flatten block shape
        d = mu_node.tensor.numel()
        mu_shape = mu_node.shape
        
        # Get Theta tensor (prior precision diagonal)
        theta = self.compute_theta_tensor(block_idx)  # Shape: mu_shape
        theta_flat = theta.flatten()  # Shape: (d,) - just keep as vector
        
        # Forward pass through μ-MPO to set up network state with all samples
        mu_output = self.forward_mu(X, to_tensor=False)
        
        # Set output_labels to match actual output
        self.mu_mpo.output_labels = tuple(mu_output.dim_labels)
        
        # Prepare y with proper shape to match output dimensions
        # Output has shape (S, r0, r2) so y should be (S, 1, 1)
        y_expanded = y
        for _ in range(len(mu_output.shape) - 1):  # Add dims for each non-sample dim
            y_expanded = y_expanded.unsqueeze(-1)
        
        # Compute J_μ terms
        # For the b term: Σₙ yₙ · J_μ(xₙ)
        sum_y_dot_J_mu = self.mu_mpo.get_b(mu_node, y_expanded)
        
        # For the A term: Σₙ J_μ(xₙ) ⊗ J_μ(xₙ) - outer product without hessian
        J_mu_outer = self._compute_mu_jacobian_outer(mu_node, self.mu_mpo, mu_output.shape)
        
        # Flatten results
        sum_y_dot_J_mu_flat = sum_y_dot_J_mu.flatten()  # Shape: (d,)
        n_dims = len(mu_shape)
        J_mu_outer_flat = J_mu_outer.flatten(0, n_dims-1).flatten(1, -1)  # Shape: (d, d)
        
        # Compute J_Σ term for Σ-MPO
        # J_Σ is just the Jacobian contracted over samples: Σₙ J_Σ(xₙ)
        # This is the b term from get_b for Σ-MPO
        sigma_output = self.forward_sigma(X, to_tensor=False)
        self.sigma_mpo.output_labels = tuple(sigma_output.dim_labels)
        
        # Create y_sigma matching sigma output shape (all ones to get Jacobian sum)
        y_sigma = torch.ones(sigma_output.shape, dtype=mu_node.tensor.dtype, device=mu_node.tensor.device)
        
        # Get J_Σ term using get_b - returns tensor with Σ-node shape
        J_sigma_sum = self.sigma_mpo.get_b(sigma_node, y_sigma)
        
        # Convert J_Σ from Σ-block shape to (d, d) matrix using existing method
        # J_sigma_sum has shape like (d1o, d1i, d2o, d2i, ...)
        # We need to reshape it to (d, d) where d = d1*d2*...
        # Use the same logic as _sigma_to_matrix but for the Jacobian
        sigma_matrix = self._sigma_to_matrix(block_idx)  # Get current Σ as (d, d)
        # J_sigma_sum should have same structure, reshape it the same way
        # Actually J_sigma_sum already has the right shape, just need to flatten properly
        
        # Get permutation to convert to matrix form
        outer_indices, inner_indices = self._get_sigma_to_mu_permutation(block_idx)
        perm = outer_indices + inner_indices
        J_sigma_permuted = J_sigma_sum.permute(*perm)
        sum_J_sigma = J_sigma_permuted.reshape(d, d)
        
        # Compute covariance inverse: Σ^⁻¹ = -E[τ] Σₙ [J_Σ(xₙ) + J_μ(xₙ) ⊗ J_μ(xₙ)] - diag(Θ)
        # Compute without diag first
        sigma_inv = -E_tau * (sum_J_sigma + J_mu_outer_flat)
        
        # Add diagonal terms: -diag(Θ) and regularization
        # Avoid creating full diagonal matrix, just subtract from diagonal
        eps = 1e-6
        diagonal_correction = theta_flat + eps
        sigma_inv.diagonal().sub_(diagonal_correction)
        
        # Compute covariance: Σ = (Σ^⁻¹)^⁻¹
        sigma_cov = torch.inverse(sigma_inv)
        
        # Compute mean: μ = -E[τ] Σ Σₙ yₙ · J_μ(xₙ)
        mu_flat = -E_tau * torch.matmul(sigma_cov, sum_y_dot_J_mu_flat)
        
        # Reshape back to block shape
        mu_new = mu_flat.reshape(mu_shape)
        
        # Update μ-block
        mu_node.tensor = mu_new
        
        # Update Σ-block
        # Convert sigma_cov (d, d) back to Σ-block shape
        sigma_new = self._matrix_to_sigma(block_idx, sigma_cov)
        sigma_node.tensor = sigma_new
        
        # Reset stacks after update
        self.mu_mpo.reset_stacks()
        self.sigma_mpo.reset_stacks()
    
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

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-6,
        trim_threshold: Optional[float] = None,
        verbose: bool = True
    ) -> None:
        """
        Fit the Bayesian MPO using coordinate ascent variational inference.
        
        Update routine:
        1. Update all blocks (μ, Σ parameters)
        2. Update all bond distributions (Gamma parameters)
        3. Update τ (precision parameter)
        4. Trim (optional)
        5. Repeat until convergence
        
        Args:
            X: Input data, shape (S, feature_dims)
            y: Target data, shape (S,)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance (not implemented yet - always runs max_iter)
            trim_threshold: If provided, trim bonds with E[X] < threshold after each iteration
            verbose: Print progress
            
        Example:
            bmpo.fit(X_train, y_train, max_iter=50, verbose=True)
        """
        if verbose:
            print('='*70)
            print('BAYESIAN MPO COORDINATE ASCENT')
            print('='*70)
            print(f'Data: {X.shape[0]} samples, {X.shape[1]} features')
            print(f'Blocks: {len(self.mu_nodes)}')
            print(f'Max iterations: {max_iter}')
            if trim_threshold is not None:
                print(f'Trim threshold: {trim_threshold}')
            print('='*70)
            print()
        
        for iteration in range(max_iter):
            if verbose:
                print(f'Iteration {iteration + 1}/{max_iter}')
            
            # Step 1: Update all blocks (μ, Σ)
            if verbose:
                print('  Updating blocks...', end='')
            for block_idx in range(len(self.mu_nodes)):
                self.update_block_variational(block_idx, X, y)
            if verbose:
                print(' ✓')
            
            # Step 2: Update all bond distributions (Gamma parameters)
            if verbose:
                print('  Updating bonds...', end='')
            # Get all bond labels (rank labels)
            bond_labels = list(self.mu_mpo.distributions.keys())
            for label in bond_labels:
                self.update_bond_variational(label)
            if verbose:
                print(f' ✓ ({len(bond_labels)} bonds)')
            
            # Step 3: Update τ (precision)
            if verbose:
                print('  Updating τ...', end='')
            self.update_tau_variational(X, y)
            tau_mean = self.get_tau_mean().item()
            if verbose:
                print(f' ✓ (E[τ] = {tau_mean:.4f})')
            
            # Step 4: Trim (optional)
            if trim_threshold is not None:
                if verbose:
                    print('  Trimming...', end='')
                # Build threshold dict for all bonds
                trim_thresholds = {label: trim_threshold for label in bond_labels}
                self.trim(trim_thresholds)
                if verbose:
                    print(' ✓')
            
            # TODO: Compute ELBO for convergence check
            # For now, just run for max_iter iterations
            
            if verbose:
                print()
        
        if verbose:
            print('='*70)
            print('TRAINING COMPLETE')
            print('='*70)
    
    def update_tau_variational(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        """
        Variational update for τ parameters (coordinate ascent).
        
        Updates q(τ) ~ Gamma(α_q, β_q) based on data (X, y).
        
        Update formulas:
        - α_q = α_p + S/2
        - β_q = β_p + Σ_s[y_s · μ(x_s)] + 1/2 * Σ_s[Σ(x_s)]
        
        where:
        - S is the number of samples
        - α_p, β_p are prior hyperparameters
        - μ(x_s) is μ-MPO forward pass for sample s (scalar output)
        - Σ(x_s) is Σ-MPO forward pass for sample s (scalar output)
        
        Args:
            X: Input data, shape (S, feature_dims) where S is number of samples
            y: Target data, shape (S,) or (S, 1) for scalar outputs
        """
        S = X.shape[0]  # Number of samples
        
        # Update α_q = α_p + S/2
        alpha_q = self.prior_tau_alpha + S / 2.0
        
        # Compute β_q = β_p + Σ_s[y_s · μ(x_s)] + 1/2 * Σ_s[Σ(x_s)]
        
        # Term 1: β_p (prior)
        beta_q = self.prior_tau_beta.clone()
        
        # Term 2: Σ_s[y_s · μ(x_s)]
        # Forward pass through μ-MPO for all samples at once
        mu_output_result = self.forward_mu(X, to_tensor=True)  # Returns Tensor when to_tensor=True
        # Type narrowing: when to_tensor=True, result is Tensor
        assert isinstance(mu_output_result, torch.Tensor)
        mu_flat = mu_output_result.reshape(S, -1)  # (S, output_dim)
        y_flat = y.reshape(S, -1)  # (S, output_dim)
        
        # Dot product: sum over all samples and output dimensions
        beta_q = beta_q + torch.sum(y_flat * mu_flat)
        
        # Term 3: 1/2 * Σ_s[Σ(x_s)]
        # Forward pass through Σ-MPO for all samples at once
        sigma_output_result = self.forward_sigma(X, to_tensor=True)  # Returns Tensor when to_tensor=True
        # Type narrowing: when to_tensor=True, result is Tensor
        assert isinstance(sigma_output_result, torch.Tensor)
        
        # Sum over all samples and output dimensions (for scalar output, just sum all)
        beta_q = beta_q + 0.5 * torch.sum(sigma_output_result)
        
        # Update the parameters
        self.update_tau(alpha=alpha_q, beta=beta_q)

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
    
    def forward_sigma(self, x: Optional[torch.Tensor] = None, to_tensor: bool = False) -> Union[TensorNode, torch.Tensor]:
        """
        Forward pass through Σ-MPO.
        
        Σ-MPO represents the correlation/variation structure E[f(X) ⊗ f(X)^T].
        If x is provided, contracts Σ-MPO with input x (sets both 'o' and 'i' inputs to x).
        If x is None, uses current input node values.
        
        Args:
            x: Input data (optional). If provided, both outer and inner inputs are set to x.
            to_tensor: If True, return tensor instead of TensorNode
            
        Returns:
            Output from Σ-MPO (correlation structure)
        """
        return self.sigma_mpo.forward(x, to_tensor=to_tensor)
    
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
        
        # Initialize prior hyperparameters (same structure as q, but separate values)
        # These act as the prior p(θ) with the same factorization as q(θ)
        self._initialize_prior_hyperparameters()
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
        
        # Flatten μ block: E_q[W_i]
        mu_flat = mu_node.tensor.flatten()
        d = mu_flat.numel()
        
        # Convert Σ block to (d, d) matrix using label-based permutation
        sigma_matrix = self._sigma_to_matrix(block_idx)
        
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
    
    def compute_expected_log_prior(self) -> torch.Tensor:
        """
        Compute E_q[log p(θ)] where p(θ) is the prior.
        
        The prior factorizes as:
        p(θ) = p(τ) × ∏ᵢ p(Wᵢ) × ∏_modes ∏_indices p(mode_param)
        
        Therefore:
        E_q[log p(θ)] = E_q[log p(τ)] + Σᵢ E_q[log p(Wᵢ)] + Σ_modes Σ_indices E_q[log p(mode_param)]
        
        Returns:
            E_q[log p(θ)] as a scalar tensor
        """
        log_p_total = torch.tensor(0.0, dtype=self.tau_alpha.dtype, device=self.tau_alpha.device)
        
        # 1. E_q[log p(τ)]
        log_p_tau = self._expected_log_prior_tau()
        log_p_total = log_p_total + log_p_tau
        
        # 2. E_q[log p(Wᵢ)] for each block
        for i in range(len(self.mu_nodes)):
            log_p_block = self._expected_log_prior_block(i)
            log_p_total = log_p_total + log_p_block
        
        # 3. E_q[log p(mode_params)] for all modes
        log_p_modes = self._expected_log_prior_modes()
        log_p_total = log_p_total + log_p_modes
        
        return log_p_total
    
    def _expected_log_prior_tau(self) -> torch.Tensor:
        """
        Compute E_q[log p(τ)] where p(τ) ~ Gamma(α₀, β₀).
        
        Uses the modular expected_log_prob method from GammaDistribution.
        
        Returns:
            E_q[log p(τ)]
        """
        # Simply call the distribution's method with prior parameters
        return self.tau_distribution.expected_log_prob(
            concentration_p=self.prior_tau_alpha,
            rate_p=self.prior_tau_beta
        )
    
    def _expected_log_prior_block(self, block_idx: int) -> torch.Tensor:
        """
        Compute E_q[log p(Wᵢ)] where p(Wᵢ) ~ N(0, Σ₀).
        
        Uses the modular expected_log_prob method from MultivariateGaussianDistribution.
        
        Args:
            block_idx: Index of the block
            
        Returns:
            E_q[log p(Wᵢ)]
        """
        # Get the q-distribution for this block
        q_block = self.get_block_q_distribution(block_idx)
        
        # Prior mean is 0
        mu_p = torch.zeros_like(q_block.loc)
        
        # Prior covariance Σ₀
        sigma_p = self.prior_block_sigma0[block_idx]
        
        # Use the modular method
        return q_block.expected_log_prob(
            loc_p=mu_p,
            covariance_matrix_p=sigma_p
        )
    
    def _expected_log_prior_modes(self) -> torch.Tensor:
        """
        Compute E_q[log p(bond_params)] for all bonds.
        
        Uses the modular expected_log_prob method from GammaDistribution.
        Each bond has N Gamma distributions (one per index).
        
        Returns:
            Sum of E_q[log p(bond_param)] over all bonds and indices
        """
        log_p_bonds_total = torch.tensor(0.0, dtype=self.tau_alpha.dtype, device=self.tau_alpha.device)
        
        for label, prior_params in self.prior_bond_params.items():
            # Get current variational distributions
            gammas_q = self.mu_mpo.get_gamma_distributions(label)
            if gammas_q is None:
                continue
            
            # Get prior parameters (unified structure)
            concentration0 = prior_params['concentration0']
            rate0 = prior_params['rate0']
            
            # Compute E_q[log p(θ)] for each index in this bond using modular method
            for i, gamma_q in enumerate(gammas_q):
                log_p_theta = gamma_q.expected_log_prob(
                    concentration_p=concentration0[i],
                    rate_p=rate0[i]
                )
                log_p_bonds_total = log_p_bonds_total + log_p_theta
        
        return log_p_bonds_total
    
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
