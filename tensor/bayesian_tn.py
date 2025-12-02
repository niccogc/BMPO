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
        sigma_tn: qtn.TensorNetwork,
        input_indices: Dict[str, List[str]],
        learnable_tags: List[str],
        tau_alpha: Optional[torch.Tensor] = None,
        tau_beta: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize Bayesian Tensor Network.
        
        Generic constructor - works with ANY tensor network topology!
        The builder/structure comes from outside.
        
        Args:
            mu_tn: Mean tensor network (parameters only, no inputs)
            sigma_tn: Covariance tensor network (doubled indices, no inputs)
                     Use bayesian_tn_builder.create_sigma_network() to create from mu_tn
            input_indices: Dict mapping input names to indices they contract with
                          e.g., {'features': ['p1', 'p2', 'p3']} for polynomial MPO
            learnable_tags: List of tensor tags that are learnable
            tau_alpha: Initial α for τ ~ Gamma(α, β) (noise precision)
            tau_beta: Initial β for τ ~ Gamma(α, β)
            device: PyTorch device
            dtype: PyTorch dtype
            
        Example:
            >>> import quimb.tensor as qtn
            >>> from tensor.bayesian_tn_builder import create_sigma_network
            >>> 
            >>> # Create mu network (any structure!)
            >>> A = qtn.Tensor(data=..., inds=('x', 'r1'), tags='A')
            >>> B = qtn.Tensor(data=..., inds=('r1', 'y'), tags='B')
            >>> mu_tn = qtn.TensorNetwork([A, B])
            >>> 
            >>> # Create sigma network
            >>> sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
            >>> 
            >>> # Create Bayesian TN
            >>> btn = BayesianTensorNetwork(
            >>>     mu_tn=mu_tn,
            >>>     sigma_tn=sigma_tn,
            >>>     input_indices={'data': ['x']},
            >>>     learnable_tags=['A', 'B']
            >>> )
        """
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        self.learnable_tags = list(learnable_tags)
        self.input_indices = input_indices
        
        # Wrap networks
        self.mu_network = QuimbTensorNetwork(
            tn=mu_tn,
            learnable_tags=learnable_tags,
            input_tags=[],
            device=self.device,
            dtype=self.dtype
        )
        
        self.sigma_network = QuimbTensorNetwork(
            tn=sigma_tn,
            learnable_tags=[tag + '_sigma' for tag in learnable_tags],
            input_tags=[],
            distributions={},
            device=self.device,
            dtype=self.dtype
        )
        
        # τ distribution
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
        Compute projection (Jacobian) for a node.
        
        This is ∂(network_output) / ∂(node_i) = T_i
        
        NEW APPROACH:
        - Remove target parameter node from network
        - Combine remaining parameters with ALL input samples
        - Return tensor with shape (batch_size, *node_shape)
        
        Args:
            node_tag: Tag of the parameter node to compute projection for
            network_type: 'mu' or 'sigma'
            inputs: Dictionary mapping input names to batched input data
                   Shape: (batch_size, features)
            
        Returns:
            Projection tensor: (batch_size, *node_shape)
            This is the Jacobian - when contracted with the node, gives output
        """
        if inputs is None:
            raise ValueError("Inputs must be provided for projection computation")
        
        network = self.mu_network if network_type == 'mu' else self.sigma_network
        tn = network.mu_tn
        
        # Get target node
        target_tag = node_tag if network_type == 'mu' else node_tag + '_sigma'
        target_tensor = tn[target_tag]  # type: ignore
        target_shape = target_tensor.shape  # type: ignore
        target_inds = target_tensor.inds  # type: ignore
        
        # Get batch size from first input
        first_input = list(inputs.values())[0]
        batch_size = first_input.shape[0]
        
        # Process each sample separately to get Jacobian
        projections = []
        
        for i in range(batch_size):
            # Get single sample
            sample_inputs = {k: v[i] for k, v in inputs.items()}
            
            # Create network WITHOUT the target node
            projection_tensors = []
            for tid, tensor in tn.tensor_map.items():
                tag = list(tensor.tags)[0] if tensor.tags else tid
                
                # Skip the target node
                if tag == target_tag:
                    continue
                
                projection_tensors.append(tensor)
            
            # Add input tensors
            # IMPORTANT: Create SEPARATE tensor for EACH index (for polynomial features)
            for input_name, input_data in sample_inputs.items():
                if input_name not in self.input_indices:
                    continue
                
                contract_indices = self.input_indices[input_name]
                
                # Convert to numpy
                if isinstance(input_data, torch.Tensor):
                    input_data_np = input_data.detach().cpu().numpy()
                else:
                    input_data_np = np.asarray(input_data)
                
                # Create SEPARATE input tensor for EACH index
                for idx in contract_indices:
                    input_tensor = qtn.Tensor(
                        data=input_data_np,
                        inds=(idx,),  # Single index
                        tags=f'input_{input_name}_{idx}'
                    )  # type: ignore
                    
                    projection_tensors.append(input_tensor)
            
            # Create projection network and contract
            if not projection_tensors:
                # Edge case: only one node, return identity
                identity = torch.eye(int(np.prod(target_shape)), device=self.device, dtype=self.dtype)
                identity = identity.reshape(*target_shape, *target_shape)
                return identity
            
            proj_tn = qtn.TensorNetwork(projection_tensors)
            
            # Contract, keeping target indices free
            # The output should have the indices that would connect to the target node
            try:
                result = proj_tn.contract(output_inds=target_inds, optimize='auto-hq')  # type: ignore
            except:
                try:
                    result = proj_tn.contract(output_inds=target_inds, optimize='greedy')  # type: ignore
                except:
                    result = proj_tn.contract(output_inds=target_inds)  # type: ignore
            
            # Convert to PyTorch
            result_array = result.data if hasattr(result, 'data') else result  # type: ignore
            
            if isinstance(result_array, np.ndarray):
                result_tensor = torch.from_numpy(result_array).to(device=self.device, dtype=self.dtype)
            else:
                result_tensor = torch.as_tensor(result_array, device=self.device, dtype=self.dtype)
            
            projections.append(result_tensor)
        
        # Stack to get (batch_size, *node_shape)
        return torch.stack(projections)
    
    def forward_mu(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward through μ network."""
        return self._forward(self.mu_network.mu_tn, inputs, index_suffix='')
    
    def _forward_single_sample(self, tn: qtn.TensorNetwork, inputs: Dict[str, torch.Tensor], index_suffix: str) -> torch.Tensor:
        """
        Forward for one sample.
        
        Args:
            tn: TensorNetwork
            inputs: Single sample inputs
            index_suffix: '' for mu, 'o' for sigma outer, 'i' for sigma inner
        """
        tn_copy = tn.copy()
        
        # Add input tensors
        for input_name, input_data in inputs.items():
            if input_name not in self.input_indices:
                continue
            
            contract_indices = self.input_indices[input_name]
            input_data_np = input_data.detach().cpu().numpy() if isinstance(input_data, torch.Tensor) else np.asarray(input_data)
            
            # For sigma: add BOTH 'o' and 'i' versions
            if index_suffix:  # sigma network
                for idx in contract_indices:
                    # Outer
                    tn_copy &= qtn.Tensor(data=input_data_np, inds=(idx + 'o',), tags=f'input_{idx}_o')  # type: ignore
                    # Inner  
                    tn_copy &= qtn.Tensor(data=input_data_np, inds=(idx + 'i',), tags=f'input_{idx}_i')  # type: ignore
            else:  # mu network
                for idx in contract_indices:
                    tn_copy &= qtn.Tensor(data=input_data_np, inds=(idx,), tags=f'input_{idx}')  # type: ignore
        
        # Contract with hyper-index handling
        from collections import Counter
        all_inds = [ind for t in tn_copy.tensor_map.values() for ind in t.inds]  # type: ignore
        ind_counts = Counter(all_inds)
        hyper_inds = [ind for ind, count in ind_counts.items() if count > 2]
        
        if hyper_inds:
            output_inds = tuple([ind for ind, count in ind_counts.items() if count == 1 or count > 2])
            try:
                result = tn_copy.contract(output_inds=output_inds, optimize='auto-hq')
            except:
                result = tn_copy.contract(output_inds=output_inds, optimize='greedy')
        else:
            try:
                result = tn_copy.contract(optimize='auto-hq')
            except:
                result = tn_copy.contract(optimize='greedy')
        
        # Convert to tensor
        result_array = result.data if hasattr(result, 'data') else result  # type: ignore
        if isinstance(result_array, np.ndarray):
            result_tensor = torch.from_numpy(result_array).to(device=self.device, dtype=self.dtype)
        elif isinstance(result_array, (int, float, np.number)):
            result_tensor = torch.tensor(float(result_array), device=self.device, dtype=self.dtype)
        else:
            result_tensor = torch.from_numpy(np.asarray(result_array)).to(device=self.device, dtype=self.dtype)
        
        return result_tensor.squeeze()
    
    def forward_sigma(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward through Σ network."""
        return self._forward(self.sigma_network.mu_tn, inputs, index_suffix='o')

    def _forward(self, tn: qtn.TensorNetwork, inputs: Dict[str, torch.Tensor], index_suffix: str) -> torch.Tensor:
        """
        Generic batched forward pass.
        
        Args:
            tn: The TensorNetwork (mu or sigma)
            inputs: Input data
            index_suffix: '' for mu, 'o'/'i' for sigma
        """
        first_input = list(inputs.values())[0]
        is_batched = first_input.dim() > 1
        
        if is_batched:
            batch_size = first_input.shape[0]
            outputs = []
            for i in range(batch_size):
                sample_inputs = {k: v[i] for k, v in inputs.items()}
                out = self._forward_single_sample(tn, sample_inputs, index_suffix)
                outputs.append(out)
            return torch.stack(outputs)
        else:
            return self._forward_single_sample(tn, inputs, index_suffix)
    
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
