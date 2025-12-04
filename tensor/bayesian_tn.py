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
        output_indices: List[str],
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
            sigma_tn: Covariance tensor network (doubled indices except outputs, no inputs)
                     Use bayesian_tn_builder.create_sigma_network() to create from mu_tn
            input_indices: Dict mapping input names to indices they contract with
                          e.g., {'features': ['p1', 'p2', 'p3']} for polynomial MPO
            output_indices: List of indices that are outputs (not doubled in sigma)
                          e.g., ['y'] for scalar output, ['y1', 'y2'] for multi-output
            learnable_tags: List of tensor tags that are learnable
            tau_alpha: Initial α for τ ~ Gamma(α, β) (noise precision)
            tau_beta: Initial β for τ ~ Gamma(α, β)
            device: PyTorch device
            dtype: PyTorch dtype
            
        Example:
            >>> import quimb.tensor as qtn
            >>> from tensor.bayesian_tn_builder import create_sigma_network
            >>> 
            >>> # Create mu network: A(x, r1) -- B(r1, y)
            >>> A = qtn.Tensor(data=..., inds=('x', 'r1'), tags='A')
            >>> B = qtn.Tensor(data=..., inds=('r1', 'y'), tags='B')
            >>> mu_tn = qtn.TensorNetwork([A, B])
            >>> 
            >>> # Create sigma network (y is NOT doubled)
            >>> sigma_tn = create_sigma_network(mu_tn, ['A', 'B'], output_indices=['y'])
            >>> 
            >>> # Create Bayesian TN
            >>> btn = BayesianTensorNetwork(
            >>>     mu_tn=mu_tn,
            >>>     sigma_tn=sigma_tn,
            >>>     input_indices={'data': ['x']},
            >>>     output_indices=['y'],
            >>>     learnable_tags=['A', 'B']
            >>> )
        """
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        # TODO: YOU SHOULD PASS THE not_learnable_tags, AND FIND THE LEARNABLE BY EXCLUSION.
        self.learnable_tags = list(learnable_tags)
        self.input_indices = input_indices
        self.output_indices = list(output_indices)
        
        # Wrap networks
        # 
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
        # TODO: INITIALIZATION FROM USER, DEFAULT RANDOM UNIFORM.
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
        
        This is the environment tensor: when contracted with the node, gives network output.
        
        APPROACH:
        - Remove target node from network
        - Add input tensors
        - Contract keeping target's SHARED indices free (using output_inds)
        - Process each sample in batch separately
        
        Args:
            node_tag: Tag of the parameter node to compute projection for
            network_type: 'mu' or 'sigma'
            inputs: Dictionary mapping input names to batched input data
                   Shape: (batch_size, features)
            
        Returns:
            Projection tensor: (batch_size, *node_shape)
            Shape matches the full node shape including boundaries.
            Dummy nodes (ones) are added for boundary indices to preserve shape.
            
        Example:
            Node A has shape (r0=1, p1=2, r1=2) where r0 is a boundary.
            Projection will have shape (batch, r0=1, p1=2, r1=2) - full shape.
        """
        if inputs is None:
            raise ValueError("Inputs must be provided for projection computation")
        # TODO: THIS HAS TO BE REFRACTORED IN ONE FUNCTION WHERE YOU PASS THE NETWORK AND THE TARGET_TAG, SO IT WORKS FOR MU AND SIGMA IN THE SAMEWAY.
        network = self.mu_network if network_type == 'mu' else self.sigma_network
        # TODO: WHY WE COPY THE MU_TN IN THE SIGMA? CANT WE JUST TAKE IT FROM SELF.MU_NETWORK? WITHOUT WASTING SPACE.
        tn = network.mu_tn
        
        # Get target node
        # TODO: SAME AS TO DO ABOVE, REFRACTORED TO BE PASSED AS ARGUMENT.
        target_tag = node_tag if network_type == 'mu' else node_tag + '_sigma'
        target_tensor = tn[target_tag]  # type: ignore[index]
        target_inds = target_tensor.inds  # type: ignore[union-attr]
        
        # Get batch size
        first_input = list(inputs.values())[0]
        batch_size = first_input.shape[0] if first_input.dim() > 1 else 1
        is_batched = first_input.dim() > 1
        
        if not is_batched:
            # Single sample
            return self._compute_projection_single_sample(tn, target_tag, target_inds, inputs, network_type)
        
        # Batch: process each sample separately
        projections = []

        # TODO: THIS IS SO SLOW. IT HAS TO BE PARALLELIZED! WTF TORCH HAVE FUNCTIONS OR JAX. NO WAY IT HAS TO CYCLE OVER BATCHES.
        for i in range(batch_size):
            sample_inputs = {k: v[i] for k, v in inputs.items()}
            proj = self._compute_projection_single_sample(tn, target_tag, target_inds, sample_inputs, network_type)
            projections.append(proj)
        
        return torch.stack(projections)
    
    def _compute_projection_single_sample(
        self,
        tn: qtn.TensorNetwork,
        target_tag: str,
        target_inds: tuple,
        inputs: Dict[str, torch.Tensor],
        network_type: str
    ) -> torch.Tensor:
        """
        Compute projection (Jacobian) for a single sample.
        
        The projection keeps:
        1. Target node's VARIATIONAL indices (not its output indices)
        2. ALL other nodes' output indices
        
        This is the derivative ∂output/∂node. When contracted with the node,
        it produces the full network output.
        
        Example:
            Node A(x, r1, y1) where y1 is output:
            Projection has: (x, r1, y2, y3) - NOT y1!
            Contract: env(x,r1,y2,y3) ⊗ A(x,r1,y1) → output(y1,y2,y3)
        
        Args:
            tn: The tensor network
            target_tag: Tag of target node to exclude
            target_inds: Indices of target node
            inputs: Single sample inputs (no batch dimension)
            network_type: 'mu' or 'sigma'
            
        Returns:
            Projection tensor with shape (*target_variational_inds, *other_output_inds)
        """
        # Build environment network WITHOUT target node
        env_tensors = []
        for tid, tensor in tn.tensor_map.items():
            tag = list(tensor.tags)[0] if tensor.tags else tid  # type: ignore[union-attr]
            if tag == target_tag:
                continue  # Skip target node
            
            # Convert to torch if needed (backend-agnostic, prefer torch)
            # TODO: ITS NOT IF NEEDED YOU ARE ALWAYS CONVERTING IT INTO TORCH.... WHY qtn cant just get tensor? why ou have to transofrm it in torch?
            tensor_data = tensor.data  # type: ignore[union-attr]
            if not isinstance(tensor_data, torch.Tensor):
                if isinstance(tensor_data, np.ndarray):
                    tensor_data_torch = torch.from_numpy(tensor_data).to(device=self.device, dtype=self.dtype)
                else:
                    tensor_data_torch = torch.tensor(tensor_data, device=self.device, dtype=self.dtype)
                # Create new tensor with torch data
                tensor_torch = qtn.Tensor(data=tensor_data_torch, inds=tensor.inds, tags=tensor.tags)  # type: ignore[union-attr,arg-type]
                env_tensors.append(tensor_torch)
            else:
                env_tensors.append(tensor)
        
        # Get target node
        target_node = tn[target_tag]  # type: ignore[index]
        target_shape = target_node.shape  # type: ignore[union-attr]
        
        # Separate target's variational indices from its output indices
        target_output_inds = [ind for ind in target_inds if ind in self.output_indices]
        target_variational_inds = [ind for ind in target_inds if ind not in self.output_indices]
        
        # Other output indices (from other nodes)
        other_output_inds = [ind for ind in self.output_indices if ind not in target_inds]
        
        # Find boundary variational indices (target's variational indices not in environment)
        from collections import Counter
        all_inds_env = []
        for t in env_tensors:
            all_inds_env.extend(t.inds)  # type: ignore[union-attr]
        ind_counts_env = Counter(all_inds_env)
        
        # Boundary variational indices: target's variational indices not in environment
        boundary_variational_inds = [ind for ind in target_variational_inds if ind not in ind_counts_env]
        
        # Add dummy nodes for boundary VARIATIONAL indices only
        # (NOT for target's output indices - they shouldn't be in the projection)
        for ind in boundary_variational_inds:
            idx_pos = list(target_inds).index(ind)
            dim_size = target_shape[idx_pos]  # type: ignore[index]
            
            # Create dummy data as torch tensor (backend-agnostic, prefer torch)
            dummy_data = torch.ones(dim_size, dtype=self.dtype, device=self.device)
            dummy_tensor = qtn.Tensor(
                data=dummy_data,  # type: ignore[arg-type]
                inds=(ind,),
                tags=f'dummy_{ind}'
            )
            env_tensors.append(dummy_tensor)
        
        # Add input tensors
        for input_name, input_data in inputs.items():
            if input_name not in self.input_indices:
                continue
            
            # Ensure input is torch tensor (backend-agnostic, prefer torch)
            if not isinstance(input_data, torch.Tensor):
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data).to(device=self.device, dtype=self.dtype)
                else:
                    input_tensor = torch.tensor(input_data, device=self.device, dtype=self.dtype)
            else:
                input_tensor = input_data.to(device=self.device, dtype=self.dtype)
            
            contract_indices = self.input_indices[input_name]
            
            if network_type == 'sigma':
                # For sigma: add BOTH 'o' and 'i' versions
                for idx in contract_indices:
                    env_tensors.append(qtn.Tensor(data=input_tensor, inds=(idx + 'o',), tags=f'input_{idx}_o'))  # type: ignore[arg-type]
                    env_tensors.append(qtn.Tensor(data=input_tensor, inds=(idx + 'i',), tags=f'input_{idx}_i'))  # type: ignore[arg-type]
            else:
                # For mu: single version
                for idx in contract_indices:
                    env_tensors.append(qtn.Tensor(data=input_tensor, inds=(idx,), tags=f'input_{idx}'))  # type: ignore[arg-type]
        
        # Create environment network
        env_tn = qtn.TensorNetwork(env_tensors)
        
        # Determine indices to keep free:
        # - Target's variational indices (to contract with target)
        # - Other nodes' output indices (free in result)
        # - NOT target's output indices!
        indices_to_keep = target_variational_inds + other_output_inds
        
        try:
            env_result = env_tn.contract(output_inds=tuple(indices_to_keep), optimize='greedy')  # type: ignore[call-overload]
        except:
            try:
                env_result = env_tn.contract(output_inds=tuple(indices_to_keep))  # type: ignore[call-overload]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to contract environment for {target_tag}. "
                    f"Target variational indices: {target_variational_inds}. "
                    f"Other output indices: {other_output_inds}. "
                    f"Indices to keep: {indices_to_keep}. "
                    f"Error: {e}"
                )
        
        # Convert to torch
        result_data = env_result.data if hasattr(env_result, 'data') else env_result  # type: ignore[union-attr]
        
        if isinstance(result_data, torch.Tensor):
            result_tensor = result_data.to(device=self.device, dtype=self.dtype)
        elif isinstance(result_data, np.ndarray):
            result_tensor = torch.from_numpy(result_data).to(device=self.device, dtype=self.dtype)
        elif isinstance(result_data, (int, float, np.number)):
            result_tensor = torch.tensor(float(result_data), device=self.device, dtype=self.dtype)
        else:
            result_tensor = torch.from_numpy(np.asarray(result_data)).to(device=self.device, dtype=self.dtype)
        
        return result_tensor
    
    def compute_projection_grad(
        self,
        node_tag: str,
        network_type: str = 'mu',
        inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute projection (Jacobian) using automatic differentiation.
        
        Alternative to compute_projection() - uses backend's autodiff instead of
        manual tensor contraction. Should give identical results.
        
        Computes: ∂(output) / ∂(node) for each sample in batch
        
        Args:
            node_tag: Tag of the parameter node
            network_type: 'mu' or 'sigma'
            inputs: Dictionary mapping input names to batched input data
            
        Returns:
            Projection tensor: (batch_size, *node_shape)
            Gradient of network output w.r.t. the target node
        """
        if inputs is None:
            raise ValueError("Inputs must be provided for projection computation")
        
        network = self.mu_network if network_type == 'mu' else self.sigma_network
        
        # Get target node
        target_tag = node_tag if network_type == 'mu' else node_tag + '_sigma'
        node_tensor = network.get_node_tensor(target_tag)
        
        # Get batch info
        first_input = list(inputs.values())[0]
        is_batched = first_input.dim() > 1
        batch_size = first_input.shape[0] if is_batched else 1
        
        if not is_batched:
            # Single sample
            return self._compute_projection_grad_single(node_tag, network_type, inputs)
        
        # Batch: process each sample
        projections = []
        for i in range(batch_size):
            sample_inputs = {k: v[i] for k, v in inputs.items()}
            proj = self._compute_projection_grad_single(node_tag, network_type, sample_inputs)
            projections.append(proj)
        
        return torch.stack(projections)
    
    def _compute_projection_grad_single(
        self,
        node_tag: str,
        network_type: str,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute projection for single sample using autograd.
        
        OPTIMIZED: Only tracks gradients for target node, not entire network.
        
        Args:
            node_tag: Tag of the parameter node
            network_type: 'mu' or 'sigma'
            inputs: Single sample inputs (no batch dimension)
            
        Returns:
            Gradient tensor with shape matching node shape
        """
        network = self.mu_network if network_type == 'mu' else self.sigma_network
        target_tag = node_tag if network_type == 'mu' else node_tag + '_sigma'
        
        # Get node tensor and make ONLY it require gradients
        node_tensor = network.get_node_tensor(target_tag).detach().clone()
        node_tensor.requires_grad_(True)
        
        # Temporarily replace node in network
        original_data = network.mu_tn[target_tag].data  # type: ignore[index]
        network.mu_tn[target_tag].modify(data=node_tensor)  # type: ignore[index,union-attr]
        
        try:
            # Forward pass - only node_tensor has requires_grad=True
            with torch.enable_grad():
                if network_type == 'mu':
                    output = self.forward_mu(inputs)
                else:
                    output = self.forward_sigma(inputs)
                
                # Compute gradient (only for target node)
                if output.dim() == 0:
                    output_scalar = output
                else:
                    output_scalar = output.sum()  # Sum if multiple outputs
                
                # Backward - only computes grad for node_tensor
                output_scalar.backward()
                
                # Extract gradient
                grad = node_tensor.grad
                
                if grad is None:
                    raise RuntimeError(f"Gradient is None for node {target_tag}")
                
                return grad.detach().clone()
        finally:
            # Always restore original node data
            network.mu_tn[target_tag].modify(data=original_data)  # type: ignore[index,union-attr]
    
    def forward_mu(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward through μ network."""
        return self._forward(self.mu_network.mu_tn, inputs, index_suffix='')
    
    def _forward_single_sample(self, tn: qtn.TensorNetwork, inputs: Dict[str, torch.Tensor], index_suffix: str) -> torch.Tensor:
        """
        Forward for one sample.
        
        Backend-agnostic: keeps data in native format (torch/jax/numpy).
        Quimb uses autoray for backend abstraction.
        
        Args:
            tn: TensorNetwork
            inputs: Single sample inputs (torch.Tensor or other backend)
            index_suffix: '' for mu, 'o' for sigma outer, 'i' for sigma inner
        """
        tn_copy = tn.copy()
        
        # Add input tensors - NO conversion, let quimb handle backend
        for input_name, input_data in inputs.items():
            if input_name not in self.input_indices:
                continue
            
            contract_indices = self.input_indices[input_name]
            
            # For sigma: add BOTH 'o' and 'i' versions
            if index_suffix:  # sigma network
                for idx in contract_indices:
                    # Outer - quimb accepts any array backend (numpy/torch/jax)
                    tn_copy &= qtn.Tensor(data=input_data, inds=(idx + 'o',), tags=f'input_{idx}_o')  # type: ignore[arg-type]
                    # Inner  
                    tn_copy &= qtn.Tensor(data=input_data, inds=(idx + 'i',), tags=f'input_{idx}_i')  # type: ignore[arg-type]
            else:  # mu network
                for idx in contract_indices:
                    tn_copy &= qtn.Tensor(data=input_data, inds=(idx,), tags=f'input_{idx}')  # type: ignore[arg-type]
        
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
        
        # Extract result - quimb returns Tensor or scalar
        result_data = result.data if hasattr(result, 'data') else result  # type: ignore[union-attr]
        
        # Convert to appropriate tensor type if needed
        if isinstance(result_data, torch.Tensor):
            result_tensor = result_data.to(device=self.device, dtype=self.dtype)
        elif isinstance(result_data, np.ndarray):
            result_tensor = torch.from_numpy(result_data).to(device=self.device, dtype=self.dtype)
        elif isinstance(result_data, (int, float, np.number)):
            result_tensor = torch.tensor(float(result_data), device=self.device, dtype=self.dtype)
        else:
            # For JAX or other backends, convert via numpy
            result_tensor = torch.from_numpy(np.asarray(result_data)).to(device=self.device, dtype=self.dtype)
        
        return result_tensor.squeeze()
    
    def forward_sigma(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward through Σ network."""
        return self._forward(self.sigma_network.mu_tn, inputs, index_suffix='o')

    def environment_forward(
        self,
        node_tag: str,
        environment: torch.Tensor,
        network_type: str = 'mu'
    ) -> torch.Tensor:
        """
        Forward pass using pre-computed environment (Jacobian).
        
        Computes: environment ⊗ node = output
        
        Processes each sample separately (quimb doesn't handle batches).
        
        Args:
            node_tag: Tag of the node to contract with environment
            environment: Pre-computed environment tensor from compute_projection()
                        Shape: (batch, *node_indices, *other_output_indices)
            network_type: 'mu' or 'sigma'
            
        Returns:
            Output tensor (batch, *all_output_indices)
            
        Example:
            >>> environment = model.compute_projection('B', 'mu', inputs)
            >>> output = model.environment_forward('B', environment, 'mu')
        """
        import autoray as ar
        
        network = self.mu_network if network_type == 'mu' else self.sigma_network
        tn = network.mu_tn
        
        # Get node
        target_tag = node_tag if network_type == 'mu' else node_tag + '_sigma'
        node_tensor = tn[target_tag]  # type: ignore[index]
        node_data = node_tensor.data  # type: ignore[union-attr]
        node_inds = node_tensor.inds  # type: ignore[union-attr]
        node_shape = node_tensor.shape  # type: ignore[union-attr]
        
        # Check if batched
        is_batched = environment.dim() > len(node_shape)
        
        if not is_batched:
            # Single sample: environment shape = (*node_indices, *other_output_indices)
            # Contract over node indices
            return self._environment_forward_single_sample(environment, node_data, node_inds, node_shape)
        
        # Batched: process each sample separately
        batch_size = environment.shape[0]
        results = []
        for i in range(batch_size):
            sample_env = environment[i]
            result = self._environment_forward_single_sample(sample_env, node_data, node_inds, node_shape)
            results.append(result)
        
        return torch.stack(results)
    
    def _environment_forward_single_sample(
        self,
        environment: torch.Tensor,
        node_data,
        node_inds: tuple,
        node_shape: tuple
    ) -> torch.Tensor:
        """
        Environment forward for a single sample.
        
        Environment has: (*node_variational_inds, *other_output_inds)
        Node has: (*node_variational_inds, *node_output_inds)
        Result: (*all_output_inds) in canonical order
        
        Args:
            environment: Single sample environment tensor
            node_data: Node data tensor  
            node_inds: Node indices
            node_shape: Node shape
            
        Returns:
            Output tensor with all output indices in canonical order
        """
        import quimb.tensor as qtn
        
        # Separate node's variational vs output indices
        node_output_inds = [ind for ind in node_inds if ind in self.output_indices]
        node_variational_inds = [ind for ind in node_inds if ind not in self.output_indices]
        
        # Other output indices (not from this node)
        other_output_inds = [ind for ind in self.output_indices if ind not in node_inds]
        
        # Environment has: node_variational_inds + other_output_inds
        env_inds = node_variational_inds + other_output_inds
        
        # Create quimb tensors with explicit index labels
        env_tensor = qtn.Tensor(
            data=environment,
            inds=tuple(env_inds),
            tags='env'
        )
        
        node_tensor = qtn.Tensor(
            data=node_data,
            inds=node_inds,
            tags='node'
        )
        
        # Contract: env(var_inds, other_outs) ⊗ node(var_inds, node_outs)
        # Sums over var_inds, keeps node_outs + other_outs = all output_inds
        result_tn = env_tensor & node_tensor
        result = result_tn.contract(optimize='auto')
        
        # Extract result data and indices
        result_data = result.data if hasattr(result, 'data') else result
        result_inds = result.inds if hasattr(result, 'inds') else None
        
        # Convert to PyTorch
        if isinstance(result_data, torch.Tensor):
            result_tensor = result_data.to(device=self.device, dtype=self.dtype)
        elif isinstance(result_data, np.ndarray):
            result_tensor = torch.from_numpy(result_data).to(device=self.device, dtype=self.dtype)
        else:
            result_tensor = torch.from_numpy(np.asarray(result_data)).to(device=self.device, dtype=self.dtype)
        
        # Permute to canonical output order if needed
        if result_inds is not None and len(result_inds) > 1:
            # Result has indices in quimb's order, permute to canonical order
            # Canonical order: self.output_indices
            perm = [result_inds.index(idx) for idx in self.output_indices if idx in result_inds]
            if perm != list(range(len(perm))):  # Only permute if needed
                result_tensor = result_tensor.permute(*perm)
        
        return result_tensor

    def _sigma_projection_to_matrix(
        self,
        node_tag: str,
        sigma_projection_summed: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert summed sigma projection to matrix form using mu node labels.
        
        Sigma projection has doubled indices (io, ii, jo, ji, ...).
        We need to reshape to (d, d) matrix where d is the variational dimension.
        
        Uses mu node indices to determine the correct ordering:
        - Outer indices (row): append 'o' to each mu variational index
        - Inner indices (col): append 'i' to each mu variational index
        
        Args:
            node_tag: Tag of the mu node
            sigma_projection_summed: Summed sigma projection with doubled indices
            
        Returns:
            Matrix of shape (d, d) where d = product of variational dimensions
        """
        # Get mu node info
        mu_node = self.mu_network.mu_tn[node_tag]  # type: ignore[index]
        mu_inds = mu_node.inds  # type: ignore[union-attr]
        mu_variational_inds = [ind for ind in mu_inds if ind not in self.output_indices]
        
        # Get sigma node info
        sigma_node = self.sigma_network.mu_tn[node_tag + '_sigma']  # type: ignore[index]
        sigma_inds = sigma_node.inds  # type: ignore[union-attr]
        
        # Build outer and inner indices from mu labels
        outer_inds = [ind + 'o' for ind in mu_variational_inds]
        inner_inds = [ind + 'i' for ind in mu_variational_inds]
        
        # Find positions in sigma_inds
        outer_positions = [sigma_inds.index(ind) for ind in outer_inds]
        inner_positions = [sigma_inds.index(ind) for ind in inner_inds]
        
        # Permute to group: outer indices first, then inner indices
        perm = outer_positions + inner_positions
        sigma_permuted = sigma_projection_summed.permute(*perm)
        
        # Calculate d from outer dimensions
        d = int(torch.prod(torch.tensor(sigma_permuted.shape[:len(outer_inds)])))
        
        # Reshape to (d, d)
        sigma_matrix = sigma_permuted.reshape(d, d)
        
        return sigma_matrix

    def environment_sum_forward(
        self,
        node_tag: str,
        environment_sum: torch.Tensor,
        network_type: str = 'mu'
    ) -> torch.Tensor:
        """
        Forward pass using sum of environments across samples.
        
        Computes: Σ_samples(environment_s) ⊗ node = scalar output
        
        This is optimized for computing aggregated quantities like:
        - Σ_s Σ(x_s) in tau update
        - Σ_s μ(x_s)² in tau update
        
        Much more efficient than computing sample-by-sample and summing.
        
        Args:
            node_tag: Tag of the node to contract
            environment_sum: Summed environment tensor (*node_shape)
                            = Σ_s environment_s
            network_type: 'mu' or 'sigma'
            
        Returns:
            Scalar tensor (output value summed over samples)
            
        Example:
            >>> # Tau update optimization
            >>> environments = [compute_projection('A', 'sigma', {k: v[i:i+1]}) 
            ...                 for i in range(batch_size)]
            >>> env_sum = torch.stack(environments).sum(dim=0)
            >>> sigma_sum = model.environment_sum_forward('A', env_sum, 'sigma')
            >>> # Now sigma_sum = Σ_s Σ(x_s), used directly in beta update
        """
        import autoray as ar
        
        network = self.mu_network if network_type == 'mu' else self.sigma_network
        tn = network.mu_tn
        
        # Get node
        target_tag = node_tag if network_type == 'mu' else node_tag + '_sigma'
        node_tensor = tn[target_tag]  # type: ignore[index]
        node_data = node_tensor.data  # type: ignore[union-attr]
        
        # Contract: environment_sum (*node_shape) with node (*node_shape)
        # Result: scalar
        
        # Backend-agnostic: flatten and compute dot product
        node_flat = ar.do('reshape', node_data, (-1,), like=node_data)
        env_flat = ar.do('reshape', environment_sum, (-1,), like=environment_sum)
        
        # Dot product: (d,) · (d,) = scalar
        result = ar.do('tensordot', env_flat, node_flat, axes=1, like=environment_sum)
        
        # Convert to PyTorch scalar for consistent API
        if isinstance(result, torch.Tensor):
            return result.to(device=self.device, dtype=self.dtype)
        elif isinstance(result, np.ndarray):
            return torch.from_numpy(result).to(device=self.device, dtype=self.dtype)
        else:
            return torch.tensor(float(result), device=self.device, dtype=self.dtype)

    def compute_node_update_terms(
        self,
        node_tag: str,
        inputs: Dict[str, torch.Tensor],
        y: torch.Tensor,
        E_tau: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the precision matrix and RHS vector for block variational update.
        
        Implements the update formula:
        Σ^{-1} = E[τ] * [Σ_n Σ_{other_out} J_σ(x_n) + Σ_n Σ_{other_out} J_μ(x_n) ⊗ J_μ(x_n)] + Θ
        μ = Σ * E[τ] * Σ_n Σ_{other_out} y_n * J_μ(x_n)
        
        Where:
        - J_σ(x_n) = projection of sigma network for sample n
        - J_μ(x_n) = projection of mu network for sample n
        - Sums are over samples AND output dimensions the block doesn't own
        - Result is expanded over the block's own output dimensions
        
        Args:
            node_tag: Tag of the node to update
            inputs: Batched input data (batch_size, features)
            y: Target data (batch_size, *output_dims)
            E_tau: Expected value of precision τ
            
        Returns:
            (precision_matrix, rhs_vector) where:
            - precision_matrix: shape (d, d) where d = product of all node dimensions
            - rhs_vector: shape (d,) 
            Both are ready for linear system solve: Σ^{-1} μ = rhs
        """
        # Get node information
        mu_tn = self.mu_network.mu_tn
        mu_node = mu_tn[node_tag]  # type: ignore[index]
        node_inds = mu_node.inds  # type: ignore[union-attr]
        node_shape = mu_node.shape  # type: ignore[union-attr]
        
        # Separate variational vs output indices  
        node_output_inds = [ind for ind in node_inds if ind in self.output_indices]
        node_variational_inds = [ind for ind in node_inds if ind not in self.output_indices]
        
        # Calculate dimensions
        d_var = int(torch.prod(torch.tensor([node_shape[list(node_inds).index(ind)] 
                                             for ind in node_variational_inds])))
        d_out = int(torch.prod(torch.tensor([node_shape[list(node_inds).index(ind)] 
                                             for ind in node_output_inds]))) if node_output_inds else 1
        d_total = d_var * d_out
        
        # Get projections
        # Shape: (batch, *node_variational_inds, *other_output_inds)
        proj_mu = self.compute_projection(node_tag, 'mu', inputs)
        proj_sigma = self.compute_projection(node_tag, 'sigma', inputs)
        
        # Determine dimensions to sum over: batch + other output dimensions
        n_var_dims = len(node_variational_inds)
        dims_to_sum = [0] + list(range(1 + n_var_dims, proj_mu.dim()))
        
        # 1. Compute Σ_n Σ_{other_out} J_σ(x_n)
        J_sigma_sum = proj_sigma.sum(dim=dims_to_sum)  # Shape: (*node_var_inds)
        
        # 2. Compute Σ_n Σ_{other_out} J_μ(x_n) ⊗ J_μ(x_n)
        batch_size = proj_mu.shape[0]
        
        # Flatten variational dimensions
        proj_mu_flat = proj_mu.reshape(batch_size, d_var, -1)  # (batch, d_var, n_other)
        
        # Outer product summed over batch and other outputs
        J_mu_outer = torch.zeros(d_var, d_var, device=self.device, dtype=self.dtype)
        for i in range(batch_size):
            proj_i = proj_mu_flat[i]  # (d_var, n_other)
            J_mu_outer += torch.mm(proj_i, proj_i.t())  # (d_var, d_var)
        
        # 3. Compute RHS: Σ_n Σ_{other_out} y_n * J_μ(x_n)
        # Sum projection over samples and other outputs to get (*var_dims)
        J_mu_sum = proj_mu.sum(dim=dims_to_sum)
        
        # For RHS, need to weight by y
        # Simplified: sum y over all output dims per sample, then weight projections
        rhs = torch.zeros(d_var, device=self.device, dtype=self.dtype)
        for i in range(batch_size):
            y_i_scalar = y[i].sum()  # Sum all output dimensions
            # Sum proj_mu[i] over other output dimensions
            proj_i_summed = proj_mu[i].sum(dim=list(range(n_var_dims, proj_mu[i].dim())))
            rhs += y_i_scalar * proj_i_summed.flatten()
        
        # Now expand for node's output dimensions if needed
        # J_sigma_sum: (*var_dims) → (d_var, d_var) matrix
        # J_mu_outer: already (d_var, d_var)
        
        # Flatten J_sigma_sum properly - it represents a matrix in variational space
        # Actually J_sigma from sigma network projection should already give us matrix structure
        # But we summed it, so we have (*var_dims) - need to interpret as diagonal or full?
        
        # For now, treat as providing diagonal elements
        J_sigma_matrix = torch.diag(J_sigma_sum.flatten())
        
        if d_out > 1:
            # Expand using Kronecker product: repeat variational structure for each output dim
            J_sigma_expanded = torch.kron(torch.eye(d_out, device=self.device, dtype=self.dtype), 
                                          J_sigma_matrix)
            J_mu_outer_expanded = torch.kron(torch.eye(d_out, device=self.device, dtype=self.dtype), 
                                             J_mu_outer)
            rhs_expanded = rhs.unsqueeze(1).expand(-1, d_out).flatten()
        else:
            J_sigma_expanded = J_sigma_matrix
            J_mu_outer_expanded = J_mu_outer
            rhs_expanded = rhs
        
        # Compute precision matrix (without theta - that's added externally)
        precision = E_tau * (J_sigma_expanded + J_mu_outer_expanded)
        rhs_final = E_tau * rhs_expanded
        
        return precision, rhs_final

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

    def get_node_dimensions(self, node_tag: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Get variational and output dimensions for a node.
        
        Args:
            node_tag: Tag of the node
            
        Returns:
            Tuple of (variational_shape, output_shape)
            - variational_shape: Shape of dimensions not in output_indices
            - output_shape: Shape of dimensions in output_indices
        """
        node_inds = self.mu_network.get_node_inds(node_tag)
        node_shape = self.mu_network.get_node_shape(node_tag)
        
        # Separate variational and output dimensions
        var_shape = tuple([node_shape[i] for i, idx in enumerate(node_inds) if idx not in self.output_indices])
        out_shape = tuple([node_shape[i] for i, idx in enumerate(node_inds) if idx in self.output_indices])
        
        return var_shape, out_shape

    def update_node_variational(
        self,
        node_tag: str,
        X: Optional[torch.Tensor],
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
            X: Input data (batch_size, ...) [DEPRECATED - use inputs_dict]
            y: Target data (batch_size,) or (batch_size, output_dims)
            inputs_dict: Dict mapping input tags to input tensors
        """
        batch_size = y.shape[0]
        
        # Get E[τ]
        E_tau = self.get_tau_mean()
        
        # Get variational and output dimensions for this node
        var_shape, out_shape = self.get_node_dimensions(node_tag)
        n_var_dims = len(var_shape)
        
        # Get theta tensor (prior precision from bonds)
        # Theta is ONLY for variational dimensions (no output dims in theta)
        # Has shape (var_shape, var_shape)
        theta = self.compute_theta_tensor(node_tag)
        
        # ============================================================
        # BATCH COMPUTATION WITH DYNAMIC EINSUM
        # ============================================================
        
        # Compute J_μ for ALL samples at once
        # Projection shape: (batch, ...variational_indices, ...other_output_indices)
        J_mu_all = self.compute_projection(node_tag, network_type='mu', inputs=inputs_dict)
        
        # J_mu_all has shape: (batch, ...var_dims, ...other_output_dims)
        # The last dimensions after var_dims are OTHER output dimensions (not from this block)
        n_other_output_dims = J_mu_all.ndim - 1 - n_var_dims  # -1 for batch dim
        
        # Create dynamic labels for einsum based on actual dimensions
        batch_label = 'b'
        var_labels = ''.join([chr(ord('i') + i) for i in range(n_var_dims)])  # i, j, k, ...
        other_out_labels = ''.join([chr(ord('o') + i) for i in range(n_other_output_dims)])  # o, p, q, ...
        
        # J_mu has indices: batch + var + other_outputs
        J_mu_indices = batch_label + var_labels + other_out_labels
        
        if False:  # Debug: print labels and actual node indices
            node_inds = self.mu_network.get_node_inds(node_tag)
            var_inds = [idx for idx in node_inds if idx not in self.output_indices]
            out_inds = [idx for idx in node_inds if idx in self.output_indices]
            print(f"\n[{node_tag}] EINSUM LABELS:")
            print(f"  Node indices: {node_inds}")
            print(f"  Variational indices: {var_inds}")
            print(f"  Output indices (this node): {out_inds}")
            print(f"  n_var_dims: {n_var_dims}, n_other_output_dims: {n_other_output_dims}")
            print(f"  batch_label: '{batch_label}'")
            print(f"  var_labels: '{var_labels}'")
            print(f"  other_out_labels: '{other_out_labels}'")
            print(f"  J_mu_indices: '{J_mu_indices}'")
        
        # Term 1: Σ_n y_n · J_μ(x_n)
        # J_mu has shape: (batch, var..., other_outputs...)
        # y has shape: (batch, ALL_output_dims...)
        # We need to contract over batch and ALL output dims, keeping only node's OWN output dims
        
        # Get this node's output dimensions
        _, node_out_shape = self.get_node_dimensions(node_tag)
        n_node_out_dims = len(node_out_shape)
        
        # Total output dims in y = len(self.output_indices)
        n_total_output_dims = len(self.output_indices)
        
        # Create labels for y: batch + all_outputs
        # y MUST have shape (batch, *output_dims) - not squeezed!
        all_output_labels = ''.join([chr(ord('o') + i) for i in range(n_total_output_dims)])
        y_indices = batch_label + all_output_labels
        
        # Create labels for node's own outputs (to keep in result)
        node_output_labels = ''.join([chr(ord('o') + i) for i in range(n_node_out_dims)])
        
        # Result should have: var_labels + node_output_labels
        result_labels = var_labels + node_output_labels
        
        einsum_str_1 = f'{J_mu_indices},{y_indices}->{result_labels}'
        sum_y_J_mu = torch.einsum(einsum_str_1, J_mu_all, y)
        # Result has shape: var_shape + node_out_shape
        
        if False:  # Debug: print mu einsum
            print(f"  MU TERM 1 (sum_y_J_mu):")
            print(f"    Einsum: '{einsum_str_1}'")
            print(f"    Actual: J_mu({', '.join(['batch'] + var_inds + (['other_outputs'] if n_other_output_dims > 0 else []))}), y(batch) -> ({', '.join(var_inds)})")
            print(f"    Shape: {J_mu_all.shape} -> {sum_y_J_mu.shape}")
        
        # Term 2: Σ_n J_μ(x_n) ⊗ J_μ(x_n)
        # Outer product over variational dims, sum over batch and other outputs
        # (batch, var..., other...) × (batch, var..., other...) -> (var..., var...,)
        var_labels_upper = var_labels.upper()  # Use uppercase for second copy
        einsum_str_2 = f'{J_mu_indices},{batch_label}{var_labels_upper}{other_out_labels}->{var_labels}{var_labels_upper}'
        sum_J_mu_outer = torch.einsum(einsum_str_2, J_mu_all, J_mu_all)
        # Result has shape: var_shape + var_shape
        
        if False:  # Debug: print mu outer einsum
            var_inds_prime = [ind + "'" for ind in var_inds]
            print(f"  MU TERM 2 (sum_J_mu_outer):")
            print(f"    Einsum: '{einsum_str_2}'")
            print(f"    Actual: J_mu(...), J_mu'(...) -> ({', '.join(var_inds + var_inds_prime)})")
            print(f"    Shape: {J_mu_all.shape} x {J_mu_all.shape} -> {sum_J_mu_outer.shape}")
        
        # Term 3: Σ_n J_Σ(x_n⊗x_n)
        # Compute sigma projection - SAME as mu projection but on sigma network!
        J_sigma_all = self.compute_projection(node_tag, network_type='sigma', inputs=inputs_dict)
        
        # Sigma projection has doubled variational indices: (batch, ...var_o, ...var_i, ...other_outputs)
        # Sigma must be reshaped as (var_labels + 'o', var_labels + 'i')
        # Use the SAME var_labels from mu, just append 'o' and 'i' 
        
        # Build einsum: var_labels with lowercase = 'o' copy, var_labels with uppercase = 'i' copy
        var_labels_out = var_labels  # Same as mu variational labels for 'o'
        var_labels_in = var_labels.upper()  # Uppercase for 'i'
        
        # Sigma has interleaved structure, build the indices string
        var_labels_interleaved = ''
        for i in range(n_var_dims):
            var_labels_interleaved += var_labels[i]  # 'o'
            var_labels_interleaved += var_labels_in[i]  # 'i'
        
        # J_sigma has indices: batch + interleaved_vars + other_outputs
        J_sigma_indices = batch_label + var_labels_interleaved + other_out_labels
        
        # Sum over batch and other_outputs, result: (var_o..., var_i...)
        # Output should match mu_outer: var_labels + var_labels_upper
        einsum_str_sigma = f'{J_sigma_indices}->{var_labels}{var_labels_upper}'
        sum_J_sigma = torch.einsum(einsum_str_sigma, J_sigma_all)
        # Result has shape: var_shape + var_shape
        
        if False:  # Debug: print sigma einsum
            # Use same convention as mu_outer: lowercase and UPPERCASE
            var_inds_lower = var_inds  # lowercase = 'out' copy
            var_inds_upper = [ind.upper() for ind in var_inds]  # UPPERCASE = 'in' copy
            var_inds_interleaved = []
            for i in range(len(var_inds)):
                var_inds_interleaved.extend([var_inds_lower[i], var_inds_upper[i]])
            print(f"  SIGMA TERM (sum_J_sigma):")
            print(f"    var_labels_interleaved: '{var_labels_interleaved}'")
            print(f"    Einsum: '{einsum_str_sigma}'")
            sigma_actual_str = ', '.join(['batch'] + var_inds_interleaved + (['other_outputs'] if n_other_output_dims > 0 else []))
            result_str = ', '.join(var_inds + [ind.upper() for ind in var_inds])
            print(f"    Actual: J_sigma({sigma_actual_str})")
            print(f"            -> ({result_str})")
            print(f"    Shape: {J_sigma_all.shape} -> {sum_J_sigma.shape}")
        
        # Compute Σ^{-1} = Θ + E[τ] * (sum_J_sigma + sum_J_mu_outer)
        # All tensors have shape (var_shape, var_shape) for variational dimensions only
        sigma_inv_var = E_tau * (sum_J_sigma + sum_J_mu_outer) + theta
        
        # Flatten only variational dims for Cholesky solve
        d_var = int(torch.prod(torch.tensor(var_shape)).item())
        sigma_inv_flat = sigma_inv_var.reshape(d_var, d_var)
        
        # sum_y_J_mu has shape (var_shape, node_out_shape)
        # Reshape to (d_var, *node_out_shape) for solve
        if node_out_shape:
            sum_y_J_mu_reshaped = sum_y_J_mu.reshape(d_var, *node_out_shape)
        else:
            sum_y_J_mu_reshaped = sum_y_J_mu.reshape(d_var)
        
        # Compute Σ and μ - solve for each output element independently
        try:
            # Use Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(sigma_inv_flat)
            rhs = E_tau * sum_y_J_mu_reshaped
            
            if node_out_shape:
                # Solve for each output element: (d_var, *out_shape)
                mu_result = torch.zeros_like(rhs)
                # Flatten output dims for batch solve
                n_out_elements = int(torch.prod(torch.tensor(node_out_shape)).item())
                rhs_flat = rhs.reshape(d_var, n_out_elements)
                for i in range(n_out_elements):
                    mu_result[:, i] = torch.cholesky_solve(rhs_flat[:, i:i+1], L).squeeze(1)
                mu_new = mu_result.reshape(var_shape + node_out_shape)
            else:
                mu_new = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1).reshape(var_shape)
                
            sigma_cov_flat = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback to direct inverse
            sigma_inv_reg = sigma_inv_flat + torch.eye(d_var, device=self.device, dtype=self.dtype) * 1e-12
            sigma_cov_flat = torch.inverse(sigma_inv_reg)
            
            if node_out_shape:
                mu_result = torch.zeros_like(sum_y_J_mu_reshaped)
                n_out_elements = int(torch.prod(torch.tensor(node_out_shape)).item())
                rhs_flat = sum_y_J_mu_reshaped.reshape(d_var, n_out_elements)
                for i in range(n_out_elements):
                    mu_result[:, i] = sigma_cov_flat @ (E_tau * rhs_flat[:, i])
                mu_new = mu_result.reshape(var_shape + node_out_shape)
            else:
                mu_new = (sigma_cov_flat @ (E_tau * sum_y_J_mu_reshaped)).reshape(var_shape)
        
        # Update mu node
        self.mu_network.set_node_tensor(node_tag, mu_new)
        
        # Update sigma node - reshape covariance matrix to sigma tensor format
        # sigma_cov_flat is (d_var, d_var), reshape back to (var_shape, var_shape)
        sigma_cov = sigma_cov_flat.reshape(var_shape + var_shape)
        
        # Covariance is (d, d) where d = product of variational dimensions
        # Sigma node shape: (d1_o, d1_i, d2_o, d2_i, ..., dn_o, dn_i, output_dims...)
        #
        # Strategy: 
        # 1. Reshape (d, d) to (d1, d2, ..., dn, d1, d2, ..., dn) using var_shape
        # 2. Transpose to interleave: (d1, d1, d2, d2, ..., dn, dn)
        # 3. Expand over output dimensions
        
        sigma_node_tag = node_tag + '_sigma'
        
        # Reshape covariance: (d, d) -> (d1, d2, ..., dn, d1, d2, ..., dn)
        extended_shape = var_shape + var_shape
        sigma_cov_reshaped = sigma_cov.reshape(extended_shape)
        
        # Transpose to interleave dimensions
        # Original: (d1, d2, ..., dn, d1', d2', ..., dn')
        # Target: (d1, d1', d2, d2', ..., dn, dn')
        n_dims = len(var_shape)
        perm = []
        for i in range(n_dims):
            perm.append(i)           # di from first half
            perm.append(i + n_dims)  # di' from second half
        
        sigma_new = sigma_cov_reshaped.permute(*perm)
        
        # If node has output dimensions, expand over them
        if out_shape:
            # Add singleton dimensions for each output dim, then expand
            for out_dim in out_shape:
                sigma_new = sigma_new.unsqueeze(-1)  # Add dimension at end
            # Expand to match output shape
            expand_shape = sigma_new.shape[:-len(out_shape)] + out_shape
            sigma_new = sigma_new.expand(*expand_shape)
        
        self.sigma_network.set_node_tensor(sigma_node_tag, sigma_new)
    
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
        
        # Step 1: Get diagonal of Σ from sigma network
        # The sigma node has variational indices doubled (e.g., r1o, r1i)
        # We need to extract the diagonal (where all 'o' indices match 'i' indices)
        sigma_node_tag = node_tag + '_sigma'
        sigma_tensor = self.sigma_network.get_node_tensor(sigma_node_tag)
        
        # Extract diagonal: where all paired indices match
        # For now, use forward sigma to get the diagonal directly
        # TODO: Optimize by extracting diagonal directly from sigma tensor
        # For now, use a simple approximation: mean of sigma values
        diag_sigma = sigma_tensor.abs().mean() * torch.ones_like(mu_tensor)
        
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
    
    def update_tau_variational(self, X: Optional[torch.Tensor], y: torch.Tensor, inputs_dict: Dict[str, torch.Tensor]) -> None:
        """
        Variational update for τ (noise precision) parameters.
        
        Implements Step 3 from THEORETICAL_MODEL.md:
        - α_τ = α_τ^0 + S/2
        - β_τ = β_τ^0 + 0.5 * Σ_n ||y_n - μ(x_n)||² + 0.5 * Σ_n Σ(x_n⊗x_n)
        
        OPTIMIZED VERSION (from TIPS_AND_TRICKS.md):
        - First term: Total MSE computed in batch
        - Second term: Uses batch forward for efficiency
        
        Args:
            X: Input data (batch_size, ...) [DEPRECATED - use inputs_dict]
            y: Target data (batch_size,) or (batch_size, *output_dims)
            inputs_dict: Dict mapping input tags to input tensors
        """
        S = y.shape[0]
        
        # Update α: α_τ = α_τ^0 + S/2
        alpha_new = self.prior_tau_alpha0 + S / 2.0
        
        # ================================================================
        # TERM 1: MSE = Σ_n ||y_n - μ(x_n)||²
        # ================================================================
        # Batch forward through μ network
        mu_batch = self.forward_mu(inputs_dict)  # (batch, *output_dims)
        
        # Compute MSE
        # Flatten output dimensions if multi-output
        if y.dim() > 1 and y.shape[1:] != torch.Size([]):
            # Multi-output: sum over output dimensions then samples
            mse = torch.sum((y - mu_batch) ** 2)
        else:
            # Single output: sum over samples
            mse = torch.sum((y.squeeze() - mu_batch.squeeze()) ** 2)
        
        # ================================================================
        # TERM 2: Trace = Σ_n Σ(x_n⊗x_n)
        # ================================================================
        # Forward through Σ network for all samples
        sigma_batch = self.forward_sigma(inputs_dict)  # (batch, *output_dims)
        
        # Sum over all dimensions (batch + output)
        if sigma_batch.dim() > 1:
            trace_sum = torch.sum(sigma_batch)
        else:
            trace_sum = torch.sum(sigma_batch)
        
        # ================================================================
        # UPDATE BETA
        # ================================================================
        beta_new = self.prior_tau_beta0 + 0.5 * mse + 0.5 * trace_sum
        
        # Update tau parameters
        self._tau_alpha = alpha_new
        self._tau_beta = beta_new
        self.tau_distribution.update_parameters(alpha_new, beta_new)
    
    def variational_update_iteration(
        self,
        y: torch.Tensor,
        inputs_dict: Dict[str, torch.Tensor],
        update_order: Optional[List[str]] = None
    ) -> None:
        """
        Perform one complete iteration of variational updates.
        
        This implements one iteration of the coordinate ascent VI algorithm from
        THEORETICAL_MODEL.md. The update order is:
        
        1. Update all node distributions (μ and Σ for each learnable node)
        2. Update all bond distributions (Gamma parameters)
        3. Update global noise precision (τ)
        
        Args:
            y: Target data (batch_size,) or (batch_size, *output_dims)
            inputs_dict: Dict mapping input names to input tensors
                        e.g., {'features': X} where X is (batch_size, feature_dim)
            update_order: Order to update nodes (default: all learnable nodes)
                         Useful for controlling update sequence in complex networks
        
        Example:
            >>> # Single iteration
            >>> model.variational_update_iteration(y, {'data': X})
            >>> 
            >>> # Multiple iterations
            >>> for i in range(100):
            >>>     model.variational_update_iteration(y, {'data': X})
        """
        if update_order is None:
            update_order = self.learnable_tags
        
        # ================================================================
        # STEP 1: Update all node distributions (μ and Σ)
        # ================================================================
        for node_tag in update_order:
            self.update_node_variational(node_tag, None, y, inputs_dict)
        
        # ================================================================
        # STEP 2: Update all bond distributions (Gamma parameters)
        # ================================================================
        for bond_label in self.mu_network.bond_labels:
            self.update_bond_variational(bond_label)
        
        # ================================================================
        # STEP 3: Update global noise precision (τ)
        # ================================================================
        self.update_tau_variational(None, y, inputs_dict)
    
    def fit(
        self,
        X: Optional[torch.Tensor],
        y: torch.Tensor,
        inputs_dict: Dict[str, torch.Tensor],
        max_iter: int = 100,
        tol: float = 1e-5,
        update_order: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fit the Bayesian Tensor Network using coordinate ascent variational inference.
        
        Runs multiple iterations of variational updates until convergence or max_iter.
        Each iteration performs:
        1. Update all node distributions (μ and Σ)
        2. Update all bond distributions
        3. Update tau (noise precision)
        
        Args:
            X: Input data (batch_size, ...) [DEPRECATED - use inputs_dict]
            y: Target data (batch_size,) or (batch_size, *output_dims)
            inputs_dict: Dict mapping input tags to input tensors
            max_iter: Maximum number of iterations
            tol: Convergence tolerance (not yet implemented)
            update_order: Order to update nodes (default: all learnable nodes)
            verbose: Whether to print progress
        
        Returns:
            Dictionary with training history:
            - 'tau': List of E[τ] values at each iteration
            - 'elbo': List of ELBO values (if implemented)
        
        Example:
            >>> model = BayesianTensorNetwork(...)
            >>> history = model.fit(
            >>>     X=None,
            >>>     y=y_train,
            >>>     inputs_dict={'data': X_train},
            >>>     max_iter=100
            >>> )
            >>> print(history['tau'])
        """
        if update_order is None:
            update_order = self.learnable_tags
        
        if verbose:
            print(f"Starting variational inference for {max_iter} iterations")
            print(f"Nodes to update: {update_order}")
            print(f"Bonds: {self.mu_network.bond_labels}")
            print(f"Batch size: {y.shape[0]}")
        
        # Track history
        history = {
            'tau': [],
            'tau_alpha': [],
            'tau_beta': []
        }
        
        for iteration in range(max_iter):
            # Perform one complete iteration
            self.variational_update_iteration(y, inputs_dict, update_order)
            
            # Record history
            E_tau = self.get_tau_mean().item()
            history['tau'].append(E_tau)
            history['tau_alpha'].append(self._tau_alpha.item())
            history['tau_beta'].append(self._tau_beta.item())
            
            # Print progress
            if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
                print(f"Iteration {iteration}: E[τ] = {E_tau:.6f}, "
                      f"α_τ = {self._tau_alpha.item():.2f}, "
                      f"β_τ = {self._tau_beta.item():.2f}")
        
        if verbose:
            print("Variational inference completed")
        
        return history
