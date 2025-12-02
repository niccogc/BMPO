"""
Bayesian Tensor Network using Quimb

General tensor network implementation for Bayesian variational inference.
Heavily based on BayesianMPO but supports arbitrary topologies via quimb.
"""

import torch
import quimb.tensor as qtn
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from tqdm import tqdm

from .quimb_network import QuimbTensorNetwork
from .probability_distributions import GammaDistribution, MultivariateGaussianDistribution, ProductDistribution


class BayesianTensorNetwork:
    """
    Bayesian Tensor Network with variational inference.
    
    Implements the GBTN (Generic Bayesian Tensor Network) algorithm from THEORETICAL_MODEL.md.
    Structure heavily based on BayesianMPO.
    
    Key components:
    - mu_tn: Mean tensor network (QuimbTensorNetwork)
    - sigma_tn: Covariance tensor network (squared bond dimensions)
    - tau: Global noise precision (Gamma distribution)
    - Variational updates for nodes, bonds, and tau
    """
    
    def __init__(
        self,
        mu_tn: qtn.TensorNetwork,
        learnable_tags: List[str],
        input_tags: Optional[List[str]] = None,
        tau_alpha: Optional[torch.Tensor] = None,
        tau_beta: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize Bayesian Tensor Network.
        
        Args:
            mu_tn: Quimb TensorNetwork for the mean (μ) network
            learnable_tags: List of tensor tags that are learnable
            input_tags: List of tensor tags that are inputs (fixed)
            tau_alpha: Initial alpha for τ ~ Gamma(α, β) (noise precision)
            tau_beta: Initial beta for τ ~ Gamma(α, β)
            device: PyTorch device
            dtype: PyTorch dtype
        """
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        self.learnable_tags = list(learnable_tags)
        self.input_tags = list(input_tags) if input_tags else []
        
        # Create μ network wrapper
        self.mu_network = QuimbTensorNetwork(
            tn=mu_tn,
            learnable_tags=learnable_tags,
            input_tags=input_tags,
            device=self.device,
            dtype=self.dtype
        )
        
        # Create Σ network (squared bond dimensions)
        self.sigma_network = self._create_sigma_network()
        
        # τ distribution: Gamma(α, β) for noise precision
        if tau_alpha is None or tau_beta is None:
            self._tau_alpha = torch.tensor(2.0, device=self.device, dtype=self.dtype)
            self._tau_beta = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        else:
            self._tau_alpha = tau_alpha.to(device=self.device, dtype=self.dtype)
            self._tau_beta = tau_beta.to(device=self.device, dtype=self.dtype)
        
        self.tau_distribution = GammaDistribution(
            concentration=self._tau_alpha,
            rate=self._tau_beta
        )
        
        # Initialize prior hyperparameters
        self._initialize_prior_hyperparameters()
    
    @property
    def tau_alpha(self) -> torch.Tensor:
        """Get tau alpha parameter."""
        return self._tau_alpha
    
    @property
    def tau_beta(self) -> torch.Tensor:
        """Get tau beta parameter."""
        return self._tau_beta
    
    def _create_sigma_network(self) -> QuimbTensorNetwork:
        """
        Create Sigma network with squared bond dimensions.
        
        For each learnable tensor with indices (i, j, k), create a sigma tensor
        with indices (io, ii, jo, ji, ko, ki) with squared dimensions.
        
        For each input with index x, create TWO input nodes with xo and xi.
        
        Similar to BayesianMPO._create_sigma_nodes()
        """
        sigma_tn_tensors = []
        
        for tag in self.learnable_tags:
            mu_tensor = self.mu_network.mu_tn[tag]  # type: ignore
            
            # Create doubled indices with 'o' (outer) and 'i' (inner) suffixes
            orig_inds = mu_tensor.inds  # type: ignore
            sigma_inds = []
            sigma_shape = []
            
            for ind, dim in zip(orig_inds, mu_tensor.shape):  # type: ignore
                # Add outer then inner: xo, xi
                sigma_inds.append(ind + 'o')
                sigma_inds.append(ind + 'i')
                sigma_shape.extend([dim, dim])
            
            # Initialize Sigma as small diagonal
            data = np.zeros(sigma_shape)
            # Set diagonal elements to small values
            for idx in np.ndindex(*[dim for dim in mu_tensor.shape]):  # type: ignore
                sigma_idx = []
                for i in idx:
                    sigma_idx.extend([i, i])
                data[tuple(sigma_idx)] = 0.01
            
            sigma_tensor = qtn.Tensor(
                data=data,  # type: ignore
                inds=tuple(sigma_inds),
                tags=tag + '_sigma'
            )
            sigma_tn_tensors.append(sigma_tensor)
        
        # Add input nodes - doubled for outer and inner
        for tag in self.input_tags:
            mu_input = self.mu_network.mu_tn[tag]  # type: ignore
            input_data = mu_input.data  # type: ignore
            input_inds = mu_input.inds  # type: ignore
            
            # Create outer input: replace x with xo
            inds_outer = tuple(ind + 'o' if ind not in ['batch', 's'] else ind for ind in input_inds)
            input_outer = qtn.Tensor(data=input_data, inds=inds_outer, tags=tag + '_o')  # type: ignore
            sigma_tn_tensors.append(input_outer)
            
            # Create inner input: replace x with xi  
            inds_inner = tuple(ind + 'i' if ind not in ['batch', 's'] else ind for ind in input_inds)
            input_inner = qtn.Tensor(data=input_data, inds=inds_inner, tags=tag + '_i')  # type: ignore
            sigma_tn_tensors.append(input_inner)
        
        sigma_tn = qtn.TensorNetwork(sigma_tn_tensors)
        
        # Create wrapper (no bond distributions for sigma network)
        # Input tags include both outer and inner versions
        sigma_input_tags = [tag + '_o' for tag in self.input_tags] + [tag + '_i' for tag in self.input_tags]
        
        sigma_network = QuimbTensorNetwork(
            tn=sigma_tn,
            learnable_tags=[tag + '_sigma' for tag in self.learnable_tags],
            input_tags=sigma_input_tags,
            distributions={},  # Sigma network doesn't have its own distributions
            device=self.device,
            dtype=self.dtype
        )
        
        return sigma_network
    
    def _initialize_prior_hyperparameters(self) -> None:
        """
        Initialize prior hyperparameters for all parameters.
        
        Similar to BayesianMPO._initialize_prior_hyperparameters()
        """
        # Prior for tau
        self.prior_tau_alpha0 = self._tau_alpha.clone()
        self.prior_tau_beta0 = self._tau_beta.clone()
        
        # Priors for bonds (already initialized in QuimbTensorNetwork)
        # Store references
        self.prior_bond_params = {}
        for bond in self.mu_network.bond_labels:
            dist_dict = self.mu_network.distributions[bond]
            self.prior_bond_params[bond] = {
                'alpha0': dist_dict['alpha_0'].clone(),
                'beta0': dist_dict['beta_0'].clone()
            }
    
    def get_tau_mean(self) -> torch.Tensor:
        """Get E[τ] = α / β."""
        return self._tau_alpha / self._tau_beta
    
    def compute_theta_tensor(self, node_tag: str, exclude_labels: Optional[List[str]] = None) -> torch.Tensor:
        """
        Compute theta tensor for a node.
        
        Delegates to QuimbTensorNetwork.compute_theta_tensor()
        """
        return self.mu_network.compute_theta_tensor(node_tag, exclude_labels)
    
    def compute_projection(
        self,
        node_tag: str,
        network_type: str = 'mu',
        inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute projection T_i (Jacobian) for a node.
        
        This is the key operation: ∂(network_output) / ∂(node_i)
        
        The projection is computed by contracting the entire network EXCEPT node i.
        
        Args:
            node_tag: Tag of the node to compute projection for
            network_type: 'mu' or 'sigma'
            inputs: Dictionary mapping input tags to input data
            
        Returns:
            Projection tensor with same indices as the node
        """
        network = self.mu_network if network_type == 'mu' else self.sigma_network
        tn = network.mu_tn
        
        # Get the target node
        target_tag = node_tag if network_type == 'mu' else node_tag + '_sigma'
        target_tensor = tn[target_tag]  # type: ignore
        target_inds = target_tensor.inds  # type: ignore
        
        # Collect all tensors EXCEPT the target node
        projection_tensors = []
        
        for tid, tensor in tn.tensor_map.items():
            tag = list(tensor.tags)[0] if tensor.tags else tid
            
            # Skip the target node
            if tag == target_tag:
                continue
            
            # Handle input nodes - replace with actual data if provided
            if tag in self.input_tags and inputs is not None and tag in inputs:
                input_data = inputs[tag]
                if isinstance(input_data, torch.Tensor):
                    input_data = input_data.detach().cpu().numpy()
                projection_tensors.append(
                    qtn.Tensor(data=input_data, inds=tensor.inds, tags=tag)  # type: ignore
                )
            else:
                projection_tensors.append(tensor)
        
        if not projection_tensors:
            # Edge case: only one node in network
            # Return identity-like tensor
            shape = target_tensor.shape  # type: ignore
            identity = torch.eye(int(np.prod(shape)), device=self.device, dtype=self.dtype)
            identity = identity.reshape(*shape, *shape)
            return identity
        
        # Create temporary network and contract
        projection_tn = qtn.TensorNetwork(projection_tensors)
        
        # Figure out which indices should remain as outputs
        # These are indices that appear an odd number of times (will be free after contraction)
        all_inds = []
        for t in projection_tensors:
            all_inds.extend(t.inds)  # type: ignore
        
        # Count index occurrences
        from collections import Counter
        ind_counts = Counter(all_inds)
        
        # Output indices are those that appear an odd number of times
        # (these will be uncontracted after the network contraction)
        # OR those that appear more than twice (hyper-indices)
        output_inds_list = []
        for ind, count in ind_counts.items():
            if count == 1 or count % 2 == 1 or count > 2:
                output_inds_list.append(ind)
        
        output_inds = tuple(output_inds_list)
        
        # Contract everything, keeping target indices (and hyper-indices) free
        try:
            result = projection_tn.contract(..., output_inds=output_inds, optimize='auto-hq')  # type: ignore
        except Exception as e:
            # Fallback to simpler contraction
            try:
                result = projection_tn.contract(..., output_inds=output_inds, optimize='greedy')  # type: ignore
            except Exception as e2:
                # Last resort: use default contraction
                result = projection_tn.contract(..., output_inds=output_inds)  # type: ignore
        
        # Convert to PyTorch
        result_array = result.data  # type: ignore
        
        if isinstance(result_array, np.ndarray):
            result_tensor = torch.from_numpy(result_array).to(device=self.device, dtype=self.dtype)
        else:
            # Handle scalars and other types
            result_tensor = torch.as_tensor(result_array, device=self.device, dtype=self.dtype)
        
        return result_tensor
    
    def forward_mu(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through μ network.
        
        Args:
            inputs: Dict mapping input tags to input tensors
            
        Returns:
            Network output
        """
        # Set input tensors in the network
        for tag, data in inputs.items():
            if tag in self.input_tags:
                self.mu_network.set_node_tensor(tag, data)
        
        # Contract the entire network
        tn_copy = self.mu_network.mu_tn.copy()
        
        # Check if we need to specify output_inds (for hyper-indices)
        all_inds = []
        for t in tn_copy.tensor_map.values():
            all_inds.extend(t.inds)  # type: ignore
        
        from collections import Counter
        ind_counts = Counter(all_inds)
        
        # If any index appears more than twice, we need to specify output_inds
        hyper_inds = [ind for ind, count in ind_counts.items() if count > 2]
        
        if hyper_inds:
            # For forward pass, we want all indices that appear once (open indices)
            output_inds = tuple([ind for ind, count in ind_counts.items() if count == 1])
            result = tn_copy.contract(..., output_inds=output_inds, optimize='auto-hq')
        else:
            result = tn_copy.contract(optimize='auto-hq')
        
        # Convert to PyTorch
        # Check if result is a Tensor object or a scalar
        if hasattr(result, 'data'):
            result_array = result.data  # type: ignore
        else:
            result_array = result
        
        if isinstance(result_array, np.ndarray):
            result_tensor = torch.from_numpy(result_array).to(device=self.device, dtype=self.dtype)
        elif isinstance(result_array, (int, float, np.number)):
            # Scalar number
            result_tensor = torch.tensor(float(result_array), device=self.device, dtype=self.dtype)
        else:
            # For memoryview or other types, try numpy conversion first
            result_array_np = np.asarray(result_array)
            result_tensor = torch.from_numpy(result_array_np).to(device=self.device, dtype=self.dtype)
        
        return result_tensor
    
    def forward_sigma(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through Σ network.
        
        The sigma network has doubled indices (outer and inner).
        We set the SAME input data to both outer and inner input nodes.
        
        Args:
            inputs: Dict mapping input tags to input tensors (same as for mu)
            
        Returns:
            Variance of network output
        """
        # For sigma network, set the SAME input to both outer and inner nodes
        for tag, data in inputs.items():
            if tag in self.input_tags:
                # Set to outer version
                self.sigma_network.set_node_tensor(tag + '_o', data)
                # Set to inner version (same data)
                self.sigma_network.set_node_tensor(tag + '_i', data)
        
        # Contract the sigma network
        tn_copy = self.sigma_network.mu_tn.copy()
        
        # Check for hyper-indices
        all_inds = []
        for t in tn_copy.tensor_map.values():
            all_inds.extend(t.inds)  # type: ignore
        
        from collections import Counter
        ind_counts = Counter(all_inds)
        hyper_inds = [ind for ind, count in ind_counts.items() if count > 2]
        
        if hyper_inds:
            output_inds = tuple([ind for ind, count in ind_counts.items() if count == 1])
            result = tn_copy.contract(..., output_inds=output_inds, optimize='auto-hq')
        else:
            result = tn_copy.contract(optimize='auto-hq')
        
        # Convert to PyTorch
        result_array = result.data  # type: ignore
        if isinstance(result_array, np.ndarray):
            result_tensor = torch.from_numpy(result_array).to(device=self.device, dtype=self.dtype)
        else:
            # Handle scalars and other types
            result_tensor = torch.as_tensor(result_array, device=self.device, dtype=self.dtype)
        
        return result_tensor
    
    def to(self, device: torch.device) -> 'BayesianTensorNetwork':
        """Move all components to device."""
        self.device = device
        self._tau_alpha = self._tau_alpha.to(device)
        self._tau_beta = self._tau_beta.to(device)
        self.tau_distribution.update_parameters(self._tau_alpha, self._tau_beta)
        self.mu_network.to(device)
        self.sigma_network.to(device)
        return self

    def update_node_variational(
        self,
        node_tag: str,
        X: torch.Tensor,
        y: torch.Tensor,
        inputs_dict: Dict[str, torch.Tensor]
    ) -> None:
        """
        Variational update for a node's parameters (μ and Σ).
        
        Implements Step 1 from THEORETICAL_MODEL.md:
        - Σ^i⁻¹ = Θ^{B(i)} + E[τ] Σ_n [J_Σ(x_n⊗x_n) + J_μ(x_n) ⊗ J_μ(x_n)]
        - μ^i = E[τ] Σ^i Σ_n y_n J_μ(x_n)
        
        Args:
            node_tag: Tag of the node to update
            X: Input data (batch_size, ...)
            y: Target data (batch_size,) or (batch_size, output_dims)
            inputs_dict: Dict mapping input tags to input tensors
        """
        batch_size = y.shape[0]
        
        # Get E[τ]
        E_tau = self.get_tau_mean()
        
        # Get theta tensor (prior precision from bonds)
        theta = self.compute_theta_tensor(node_tag)
        theta_flat = theta.flatten()
        d = theta_flat.numel()  # Flattened dimension
        
        # Get node shape
        node_shape = self.mu_network.get_node_shape(node_tag)
        
        # ============================================================
        # BATCH COMPUTATION (like BayesianMPO)
        # ============================================================
        
        # Compute J_μ for ALL samples at once
        # Shape: (batch, node_dims...) e.g., (batch, x, r1, batch_out)
        J_mu_all = self.compute_projection(node_tag, network_type='mu', inputs=inputs_dict)
        
        # Flatten node dimensions while keeping batch separate
        # From (batch, d1, d2, ...) to (batch, d)
        J_mu_batch = J_mu_all.reshape(batch_size, -1)
        
        if J_mu_batch.shape[1] != d:
            raise ValueError(f"Projection size mismatch: got {J_mu_batch.shape[1]}, expected {d}")
        
        # Term 1: Σ_n y_n · J_μ(x_n)
        # Contract over batch dimension: (batch, d) × (batch,) -> (d,)
        y_vec = y if y.dim() == 1 else y.squeeze(-1)
        sum_y_J_mu = torch.einsum('bd,b->d', J_mu_batch, y_vec)
        
        # Term 2: Σ_n J_μ(x_n) ⊗ J_μ(x_n)
        # Outer product contracted over batch: (batch, d) × (batch, d) -> (d, d)
        sum_J_mu_outer = torch.einsum('bd,bD->dD', J_mu_batch, J_mu_batch)
        
        # Term 3: Σ_n J_Σ(x_n⊗x_n)
        # For now, skip sigma term (will be small if Σ is small)
        # TODO: Implement proper sigma Jacobian computation
        sum_J_sigma = torch.zeros(d, d, device=self.device, dtype=self.dtype)
        
        # Compute Σ^{-1} = Θ + E[τ] * (sum_J_sigma + sum_J_mu_outer)
        sigma_inv = E_tau * (sum_J_sigma + sum_J_mu_outer)
        sigma_inv.diagonal().add_(theta_flat)  # Add diagonal theta
        
        # Compute Σ and μ
        try:
            # Use Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(sigma_inv)
            rhs = E_tau * sum_y_J_mu
            mu_flat = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
            # sigma_cov = torch.cholesky_inverse(L)  # Not needed if we don't update sigma network
        except RuntimeError:
            # Fallback to direct inverse
            sigma_inv_reg = sigma_inv + torch.eye(d, device=self.device, dtype=self.dtype) * 1e-6
            sigma_cov = torch.inverse(sigma_inv_reg)
            mu_flat = sigma_cov @ (E_tau * sum_y_J_mu)
        
        # Reshape and update mu node
        mu_new = mu_flat.reshape(node_shape)
        self.mu_network.set_node_tensor(node_tag, mu_new)
        
        # TODO: Update sigma network similarly if needed
    
    def partial_trace_update(self, node_tag: str, bond_label: str) -> torch.Tensor:
        """
        Compute partial trace for variational update of a specific bond.
        
        Based on BayesianMPO.partial_trace_update().
        
        Computes: Σ_{other indices} [diag(Σ) + μ²] × Θ_{without bond_label}
        
        Args:
            node_tag: Tag of the node
            bond_label: The bond label to keep (not sum over)
            
        Returns:
            Vector of length equal to the dimension of bond_label
        """
        # Get μ tensor
        mu_tensor = self.mu_network.get_node_tensor(node_tag)
        mu_shape = mu_tensor.shape
        
        # Get node indices
        node_inds = self.mu_network.get_node_inds(node_tag)
        
        # Check that bond_label is in this node
        if bond_label not in node_inds:
            raise ValueError(f"Bond '{bond_label}' not in node '{node_tag}'. Available: {node_inds}")
        
        # Step 1: Get diagonal of Σ (for now, approximate as zeros since we don't update sigma)
        # In full implementation, would extract diagonal from sigma network
        diag_sigma = torch.zeros_like(mu_tensor)
        
        # Step 2: Compute v = diag(Σ) + μ²
        v = diag_sigma + mu_tensor ** 2
        
        # Step 3: Get Θ excluding bond_label
        theta = self.mu_network.compute_theta_tensor(node_tag, exclude_labels=[bond_label])
        
        # Step 4: Multiply element-wise
        result = v * theta
        
        # Step 5: Sum over all dimensions except bond_label
        bond_dim_idx = node_inds.index(bond_label)
        dims_to_sum = [i for i in range(len(mu_shape)) if i != bond_dim_idx]
        output = torch.sum(result, dim=dims_to_sum)
        
        return output
    
    def update_bond_variational(self, bond_label: str) -> None:
        """
        Variational update for a bond's Gamma distribution parameters.
        
        Implements Step 2 from THEORETICAL_MODEL.md:
        - α_i = α_i^0 + |A(i)|·dim(i)/2
        - β_i = β_i^0 + 0.5 * Σ_{a∈A(i)} partial_trace_update(a, i)
        
        Based on BayesianMPO.update_bond_variational().
        
        Args:
            bond_label: Label of the bond to update
        """
        # Get bond information
        if bond_label not in self.mu_network.distributions:
            return  # Not a learnable bond
        
        bond_params = self.mu_network.distributions[bond_label]
        bond_dim = len(bond_params['alpha'])
        
        # Get nodes connected to this bond
        connected_nodes = self.mu_network.get_nodes_for_bond(bond_label)
        num_nodes = len(connected_nodes)
        
        if num_nodes == 0:
            return
        
        # Get prior parameters
        alpha_0 = bond_params['alpha_0']
        beta_0 = bond_params['beta_0']
        
        # Update alpha: α_i = α_i^0 + |A(i)|·dim(i)/2
        alpha_new = alpha_0 + (num_nodes * bond_dim) / 2.0
        
        # Update beta: β_i = β_i^0 + 0.5 * Σ_{a∈A(i)} partial_trace
        beta_new = beta_0.clone()
        
        for node_tag in connected_nodes:
            # Compute partial trace for this node
            partial_trace = self.partial_trace_update(node_tag, bond_label)
            beta_new = beta_new + 0.5 * partial_trace
        
        # Update bond parameters
        self.mu_network.update_distribution_params(bond_label, alpha_new, beta_new)
    
    def update_tau_variational(self, X: torch.Tensor, y: torch.Tensor, inputs_dict: Dict[str, torch.Tensor]) -> None:
        """
        Variational update for τ (noise precision) parameters.
        
        Implements Step 3 from THEORETICAL_MODEL.md:
        - α_τ = α_τ^0 + S/2
        - β_τ = β_τ^0 + 0.5 * Σ_n[y_n² - 2y_n·μ(x_n) + μ(x_n)² + Σ(x_n)]
        
        Args:
            X: Input data (batch_size, ...)
            y: Target data (batch_size,)
            inputs_dict: Dict mapping input tags to input tensors
        """
        S = y.shape[0]
        
        # Update α: α_τ = α_τ^0 + S/2
        alpha_new = self.prior_tau_alpha0 + S / 2.0
        
        # Update β: β_τ = β_τ^0 + 0.5 * Σ[y² - 2y·μ + μ² + Σ]
        beta_new = self.prior_tau_beta0.clone()
        
        for i in range(S):
            # Prepare single-sample inputs
            sample_inputs = {
                tag: inputs_dict[tag][i:i+1] for tag in inputs_dict.keys()
            }
            
            # Forward through μ network
            mu_i = self.forward_mu(sample_inputs)
            mu_i = mu_i.item() if mu_i.numel() == 1 else mu_i.squeeze()
            
            # Forward through Σ network (variance)
            sigma_i = self.forward_sigma(sample_inputs)
            sigma_i = sigma_i.item() if sigma_i.numel() == 1 else sigma_i.squeeze()
            
            # y_i
            y_i = y[i].item() if y[i].numel() == 1 else y[i].squeeze()
            
            # Accumulate: y² - 2y·μ + μ² + Σ
            beta_new = beta_new + 0.5 * (y_i**2 - 2*y_i*mu_i + mu_i**2 + sigma_i)
        
        # Update tau parameters
        self._tau_alpha = alpha_new
        self._tau_beta = beta_new
        self.tau_distribution.update_parameters(alpha_new, beta_new)
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        inputs_dict: Dict[str, torch.Tensor],
        max_iter: int = 100,
        tol: float = 1e-5,
        update_order: Optional[List[str]] = None
    ) -> None:
        """
        Fit the Bayesian Tensor Network using coordinate ascent variational inference.
        
        Args:
            X: Input data (batch_size, ...)
            y: Target data (batch_size,)
            inputs_dict: Dict mapping input tags to input tensors
            max_iter: Maximum number of iterations
            tol: Convergence tolerance (not yet implemented)
            update_order: Order to update nodes (default: all learnable nodes)
        """
        if update_order is None:
            update_order = self.learnable_tags
        
        print(f"Starting variational inference for {max_iter} iterations")
        print(f"Nodes to update: {update_order}")
        print(f"Bonds: {self.mu_network.bond_labels}")
        
        for iteration in range(max_iter):
            # Update each node
            for node_tag in update_order:
                self.update_node_variational(node_tag, X, y, inputs_dict)
            
            # Update bonds
            for bond in self.mu_network.bond_labels:
                self.update_bond_variational(bond)
            
            # Update tau
            self.update_tau_variational(X, y, inputs_dict)
            
            # Print progress
            if iteration % 10 == 0:
                E_tau = self.get_tau_mean()
                print(f"Iteration {iteration}: E[τ] = {E_tau.item():.6f}")
        
        print("Variational inference completed")
