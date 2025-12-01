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
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Sequence, Set, Tuple, Union, Any, Set, Tuple, Union


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
    
    def __init__(
        self, 
        input_nodes: List[TensorNode], 
        main_nodes: List[TensorNode], 
        train_nodes: Optional[List[TensorNode]] = None, 
        output_labels: Tuple[str, ...] = ('s',), 
        sample_dim: str = 's', 
        rank_labels: Optional[Union[Set[str], List[str]]] = None, 
        distributions: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
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
        
        # Build mapping from bonds to associated nodes
        self.bond_to_nodes = self._build_bond_to_nodes_mapping()
    
    def _infer_rank_labels(self) -> Set[str]:
        """Infer which dimensions are ranks from node left_labels and right_labels."""
        rank_labels: Set[str] = set()
        for node in self.main_nodes:
            rank_labels.update(node.left_labels)
            rank_labels.update(node.right_labels)
        return rank_labels
    
    def _initialize_distributions(self) -> None:
        """
        Initialize Gamma distributions for all unique dimension labels.
        
        Each dimension (bond) gets N Gamma(concentration, rate) distributions,
        where N is the dimension size. Uses unified parameterization regardless
        of whether the bond is horizontal (rank) or vertical.
        
        Note: Dimensions of size 1 are dummy indices for consistency and don't
        get Gamma distributions (not learnable).
        """
        for node in self.main_nodes:
            for label, size in zip(node.dim_labels, node.shape):
                if label not in self.distributions and size > 1:
                    dtype = node.tensor.dtype
                    
                    # Unified structure for all bonds (only if size > 1)
                    self.distributions[label] = {
                        'concentration': torch.ones(size, dtype=dtype) * 2.0,  # Default α = 2
                        'rate': torch.ones(size, dtype=dtype) * 1.0,           # Default β = 1
                        'expectation': torch.ones(size, dtype=dtype) * 2.0     # E[X] = α/β = 2
                    }

    def _build_bond_to_nodes_mapping(self) -> Dict[str, List[int]]:
        """
        Build mapping from bond labels to node indices.
        
        For each bond (dimension label), store which nodes contain that bond.
        - Horizontal bonds (rank labels): appear in left/right of consecutive nodes
        - Vertical bonds: appear in a single node
        
        Returns:
            Dict mapping bond_label -> list of node indices that share this bond
        """
        bond_to_nodes: Dict[str, List[int]] = {}
        
        for node_idx, node in enumerate(self.main_nodes):
            for label in node.dim_labels:
                if label not in bond_to_nodes:
                    bond_to_nodes[label] = []
                
                # Add this node index if not already present
                if node_idx not in bond_to_nodes[label]:
                    bond_to_nodes[label].append(node_idx)
        
        return bond_to_nodes
    
    def get_distribution_params(self, label: str) -> Optional[Dict[str, Any]]:
        """Get distribution parameters for a dimension label."""
        return self.distributions.get(label)
    
    def get_expectations(self, label: str) -> Optional[torch.Tensor]:
        """Get expectation values E[ω] or E[φ] for a dimension."""
        dist = self.distributions.get(label)
        return dist['expectation'] if dist else None
    
    def update_distribution_params(
        self, 
        label: str, 
        concentration: Optional[torch.Tensor] = None, 
        rate: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update Gamma distribution parameters for a bond.
        
        Args:
            label: Bond label
            concentration: Concentration parameters (α) for each index
            rate: Rate parameters (β) for each index
        """
        if label not in self.distributions:
            raise ValueError(f"Label '{label}' not found in distributions")
        
        dist = self.distributions[label]
        
        if concentration is not None:
            dist['concentration'] = concentration
        if rate is not None:
            dist['rate'] = rate
        
        # Update expectations using mean() from GammaDistribution objects
        self._update_expectations(label)
    
    def _update_expectations(self, label: str) -> None:
        """Update expectation values from Gamma distribution means."""
        gammas = self.get_gamma_distributions(label)
        if gammas is not None:
            expectations = torch.tensor([gamma.mean() for gamma in gammas])
            self.distributions[label]['expectation'] = expectations
    
    def get_gamma_distributions(self, label: str) -> Optional[List[GammaDistribution]]:
        """
        Get list of Gamma distribution objects for a bond.
        
        Args:
            label: Bond label
            
        Returns:
            List of GammaDistribution objects, one per index in the bond dimension
        """
        dist = self.distributions.get(label)
        if dist is None:
            return None
        
        distributions = []
        # Unified structure: Gamma(concentration, rate) for each index
        for i in range(len(dist['concentration'])):
            gamma = GammaDistribution(
                concentration=dist['concentration'][i],
                rate=dist['rate'][i]
            )
            distributions.append(gamma)
        
        return distributions
    
    def get_product_distribution(self, label: str) -> Optional[ProductDistribution]:
        """
        Get product distribution for all indices in a dimension.
        
        Returns:
            ProductDistribution of all Gamma distributions for this dimension
        """
        gammas = self.get_gamma_distributions(label)
        if gammas is None:
            return None
        return ProductDistribution(gammas)  # type: ignore

    def get_nodes_for_bond(self, label: str) -> List[int]:
        """
        Get list of node indices that share the given bond.
        
        Args:
            label: Bond label
            
        Returns:
            List of node indices (0-indexed) that contain this bond
        """
        return self.bond_to_nodes.get(label, [])
    
    def compute_theta_tensor(self, block_idx: int, exclude_labels: Optional[List[str]] = None) -> torch.Tensor:
        """
        Compute the Theta tensor for a block from expectations of Gamma distributions.
        
        Theta is the tensor product of E_q[X] for each bond in the block.
        For block with bonds [r1, d1, r2]: Theta_{ijk} = E_q[r1]_i × E_q[d1]_j × E_q[r2]_k
        
        OPTIMIZED: Uses einsum for efficient outer product computation when possible.
        
        Args:
            block_idx: Index of the block (0 to N-1)
            exclude_labels: List of bond labels to exclude (set their dimension to 1)
                           Example: exclude=['d1'] sets that dimension to 1
                           
        Returns:
            Theta tensor with shape matching the block shape, where excluded
            dimensions have size 1
            
        Example:
            # Block 2 has bonds ['r1', 'd1', 'r2'] with shape (4, 5, 4)
            theta = bmpo.compute_theta_tensor(2)  
            # Returns tensor of shape (4, 5, 4) = outer product of expectations
            
            theta_excl = bmpo.compute_theta_tensor(2, exclude_labels=['d1'])
            # Returns tensor of shape (4, 1, 4), middle dimension set to 1
        """
        if exclude_labels is None:
            exclude_labels = []
        
        node = self.main_nodes[block_idx]
        
        # Collect factors (expectation vectors) for each dimension
        factors = []
        for label, size in zip(node.dim_labels, node.shape):
            if label in exclude_labels:
                # Excluded: use vector of ones with size 1
                factor = torch.ones(1, dtype=node.tensor.dtype, device=node.tensor.device)
            else:
                # Get expectations for this bond
                if label in self.distributions:
                    factor = self.distributions[label]['expectation']
                else:
                    # Dummy index (size 1, no Gamma dist): use 1
                    factor = torch.ones(size, dtype=node.tensor.dtype, device=node.tensor.device)
            factors.append(factor)
        
        # OPTIMIZATION: Use einsum for efficient outer product
        # Generate einsum string: 'i,j,k->ijk' for 3D tensor
        num_dims = len(factors)
        
        if num_dims == 0:
            # Edge case: no dimensions
            return torch.tensor(1, dtype=node.tensor.dtype, device=node.tensor.device)
        elif num_dims == 1:
            # Single dimension: just scale
            return factors[0].unsqueeze(0) if factors[0].numel() == 1 else factors[0]
        else:
            # Multiple dimensions: use einsum
            # Create symbolic indices for each dimension
            input_indices = ','.join([chr(97 + i) for i in range(num_dims)])  # 'a,b,c,...'
            output_indices = ''.join([chr(97 + i) for i in range(num_dims)])   # 'abc...'
            einsum_str = f'{input_indices}->{output_indices}'
            
            theta = torch.einsum(einsum_str, *factors)
            
        return theta
    
    def compute_entropy(self, label: str) -> Union[torch.Tensor, None]:
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
        return product_dist.entropy()  # type: ignore
    
    def get_jacobian(self, node: TensorNode):
        """
        Compute Jacobian (J) for a node - the network contracted without that block.
        
        Args:
            node: Node to compute Jacobian for
            
        Returns:
            jacobian_node: TensorNode representing J
        """
        return self.compute_jacobian_stack(node)
    
    def trim(self, thresholds: Dict[str, float]) -> 'BMPONetwork':
        """
        Trim all nodes in the network based on expectation value thresholds.
        
        For each dimension label, keeps only indices where E[X] >= threshold.
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
            
            # Trim distribution parameters (unified structure)
            dist['concentration'] = dist['concentration'][indices]
            dist['rate'] = dist['rate'][indices]
            dist['expectation'] = dist['expectation'][indices]
        
        # Trim all node tensors
        all_nodes = self.input_nodes + self.main_nodes
        for node in all_nodes:
            self._trim_node(node, keep_indices)
        
        # Reset stacks after trimming
        self.reset_stacks()
        
        return self
    
    def _trim_node(self, node: TensorNode, keep_indices: Dict[str, torch.Tensor]) -> None:
        """Trim a node's tensor based on keep_indices for each dimension."""
        new_tensor = node.tensor
        
        for dim_idx, label in enumerate(node.dim_labels):
            if label in keep_indices:
                indices = keep_indices[label]
                new_tensor = torch.index_select(new_tensor, dim_idx, indices)
        
        node.tensor = new_tensor
    
    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> 'BMPONetwork':
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
