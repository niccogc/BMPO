# type: ignore
from typing import List, Dict, Optional, Tuple
import quimb.tensor as qt
import numpy as np

from tensor.builder import BTNBuilder

class BTN:
    def __init__(self,
                 mu: qt.TensorNetwork,
                 output_dimensions: List[str],
                 batch_dim: str = "s",
                 fixed_nodes: List[str] = None
                 ):
        """
        Bayesian Tensor Network.
        
        Args:
            mu: The mean TensorNetwork topology.
            output_dimensions: List of output indices.
            batch_dim: Batch dimension label.
            fixed_nodes: List of nodes that should not be optimized (optional).
        """
        self.mu = mu
        self.output_dimensions = output_dimensions
        self.batch_dim = batch_dim
        self.fixed_nodes = fixed_nodes if fixed_nodes else []

        # --- Initialize Builder and Model Components ---
        self.builder = BTNBuilder(self.mu, self.output_dimensions, self.batch_dim)
        
        # Build and unpack components
        # p_bonds: Priors for edges (Gamma)
        # p_nodes: Priors for nodes (Gaussian)
        # q_bonds: Posteriors for edges (Gamma)
        # q_nodes: Posteriors for nodes (Gaussian)
        # sigma: The covariance TensorNetwork
        (self.p_bonds, 
         self.p_nodes, 
         self.q_bonds, 
         self.q_nodes, 
         self.sigma) = self.builder.build_model()

    def _copy_data(self, data):
        """Helper to copy data arrays, backend-agnostic."""
        if hasattr(data, 'clone'):  # PyTorch
            return data.clone()
        elif hasattr(data, 'copy'):  # NumPy, JAX
            return data.copy()
        else:
            # Fallback
            import numpy as np
            return np.array(data)

    def _batch_forward(self, tn: qt.TensorNetwork, inputs: List[qt.Tensor], output_inds: List[str]) -> qt.Tensor:
        """Helper for forward pass, contracting a single batch of inputs."""
        full_tn = tn & inputs
        return full_tn.contract(output_inds=output_inds)

    def forward(self, tn: qt.TensorNetwork, inputs: List[List[qt.Tensor]], 
                sum_over_batch: bool = False, sum_over_output: bool = False):
        """
        Performs the batched forward pass.
        
        Args:
            tn: TensorNetwork to contract
            inputs: List of batches, where each batch is a list of input tensors
            sum_over_batch: If True, removes batch_dim from output_inds (sums over it)
            sum_over_output: If True, also removes output_dimensions from output_inds
            
        Returns:
            Result tensor with appropriate indices based on flags.
        """
        # Determine output indices based on flags
        if sum_over_output:
            if sum_over_batch:
                target_inds = []  # Sum over everything - scalar
            else:
                target_inds = [self.batch_dim]  # Keep only batch dim
        elif sum_over_batch:
            target_inds = self.output_dimensions  # Only keep output dims
        else:
            target_inds = [self.batch_dim] + self.output_dimensions  # Keep both
        
        if sum_over_batch:
            # Sum on-the-fly - don't store all batches
            result = None
            for batch_tensors in inputs:
                res = self._batch_forward(tn, batch_tensors, output_inds=target_inds)
                
                if len(target_inds) > 0:
                    res.transpose_(*target_inds)
                
                if result is None:
                    result = res
                else:
                    # Sum using quimb + operator
                    result = result + res
            
            return result
        else:
            # Concatenate batches - need to collect them
            batch_results = []
            for batch_tensors in inputs:
                res = self._batch_forward(tn, batch_tensors, output_inds=target_inds)
                
                if len(target_inds) > 0:
                    res.transpose_(*target_inds)
                
                batch_results.append(res)
            
            return self._concat_batch_results(batch_results)

    def prepare_inputs(self, input_data: Dict[str, np.ndarray], for_sigma: bool = False) -> List[qt.Tensor]:
        """
        Prepare input tensors for forward pass through the network.
        
        Args:
            input_data: Dictionary mapping input index names to data arrays.
                       Two scenarios supported:
                       1. Single input for all nodes: {'x1': array of shape (samples, features)}
                          Creates identical input nodes x1, x2, ... with same data
                       2. Separate inputs per node: {'x1': data_1, 'x2': data_2, ...}
                          Each input index gets its own data
            for_sigma: If True, doubles the inputs with '_prime' suffix for sigma network.
                      If False, creates single copy for mu network.
        
        Returns:
            List of quimb.Tensor objects ready for forward pass.
            
        Example:
            # Scenario 1: Same data for all input nodes
            input_data = {'x1': np.random.randn(10, 4)}  # 10 samples, 4 features
            mu_inputs = btn.prepare_inputs(input_data, for_sigma=False)
            # Creates: x1(s, x1, data), x2(s, x2, data), ... with same data
            
            # Scenario 2: Different data per node
            input_data = {'x1': data_1, 'x2': data_2}
            mu_inputs = btn.prepare_inputs(input_data, for_sigma=False)
            # Creates: x1(s, x1, data_1), x2(s, x2, data_2)
            
            # For sigma network (doubles inputs):
            sigma_inputs = btn.prepare_inputs(input_data, for_sigma=True)
            # Creates: x1(s, x1, data), x1_prime(s, x1_prime, data), x2(s, x2, data), x2_prime(s, x2_prime, data)
        """
        # Identify input indices from the mu network
        # Input indices are "leaf" indices that appear in only one tensor
        # (not internal bonds that connect two tensors)
        ind_count = {}
        for tensor in self.mu:
            for ind in tensor.inds:
                if ind not in self.output_dimensions and ind != self.batch_dim:
                    ind_count[ind] = ind_count.get(ind, 0) + 1
        
        # Input indices are those that appear only once (leaf nodes)
        input_indices = sorted([ind for ind, count in ind_count.items() if count == 1])
        
        # Determine if we have single input for all or separate inputs
        if len(input_data) == 1:
            # Scenario 1: Single input data for all input nodes
            single_key = list(input_data.keys())[0]
            data = input_data[single_key]
            
            # Ensure data is at least 2D (samples, features)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            batch_size, feature_dim = data.shape
            
            # Create tensors for each input index pointing to the SAME data
            tensors = []
            for input_idx in input_indices:
                # Create tensor with indices (batch_dim, input_idx)
                # All tensors point to the same data object - no copy needed
                tensor = qt.Tensor(
                    data=data,
                    inds=(self.batch_dim, input_idx),
                    tags={f'input_{input_idx}'}
                )
                tensors.append(tensor)
                
                # If for_sigma, also create the prime version
                if for_sigma:
                    prime_idx = f"{input_idx}_prime"
                    tensor_prime = qt.Tensor(
                        data=data,  # Points to same data
                        inds=(self.batch_dim, prime_idx),
                        tags={f'input_{prime_idx}'}
                    )
                    tensors.append(tensor_prime)
        else:
            # Scenario 2: Separate data for each input index
            tensors = []
            for input_idx in input_indices:
                if input_idx not in input_data:
                    raise ValueError(f"Missing data for input index '{input_idx}'. "
                                   f"Provided: {list(input_data.keys())}, Required: {input_indices}")
                
                data = input_data[input_idx]
                
                # Ensure data is at least 2D (samples, features)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                # Create tensor with indices (batch_dim, input_idx)
                tensor = qt.Tensor(
                    data=data,
                    inds=(self.batch_dim, input_idx),
                    tags={f'input_{input_idx}'}
                )
                tensors.append(tensor)
                
                # If for_sigma, also create the prime version with the same data
                if for_sigma:
                    prime_idx = f"{input_idx}_prime"
                    tensor_prime = qt.Tensor(
                        data=data,  # Points to same data
                        inds=(self.batch_dim, prime_idx),
                        tags={f'input_{prime_idx}'}
                    )
                    tensors.append(tensor_prime)
        
        return tensors

    def get_environment(self, tn: qt.TensorNetwork, target_tag: str, copy: bool = True,
                        sum_over_batch: bool = False, sum_over_output: bool = False) -> qt.Tensor:
        """
        Calculates the environment for a target tensor (single batch).
        This is the base method that works on a network with a single batch of inputs.
        
        Args:
            tn: TensorNetwork with inputs attached (single batch)
            target_tag: Tag identifying the tensor to remove
            copy: Whether to copy the network before modification
            sum_over_batch: If True, removes batch_dim from output_inds (sums over it)
            sum_over_output: If True, removes output_dimensions from output_inds (sums over outputs)
            
        Returns:
            Environment tensor with indices determined by flags.
            The "hole" indices (where the removed tensor connects) are always kept.
        """
        # 1. Create the "hole"
        if copy:
            env_tn = tn.copy()
        else:
            env_tn = tn
        env_tn.delete(target_tag)

        # 2. Determine which indices to keep based on flags
        outer_inds = list(env_tn.outer_inds())
        
        # Start with all outer indices
        final_env_inds = outer_inds.copy()
        
        # Remove batch dimension if sum_over_batch=True
        if sum_over_batch and self.batch_dim in final_env_inds:
            final_env_inds.remove(self.batch_dim)
        
        # Remove output dimensions if sum_over_output=True
        if sum_over_output:
            for out_dim in self.output_dimensions:
                if out_dim in final_env_inds:
                    final_env_inds.remove(out_dim)
        
        # If batch dimension is not in outer_inds but exists and we want to keep it
        if not sum_over_batch and self.batch_dim not in final_env_inds:
            # Check if batch dim exists in the environment
            all_inds_in_env = set()
            for tensor in env_tn:
                all_inds_in_env.update(tensor.inds)
            
            if self.batch_dim in all_inds_in_env:
                # Batch dim exists but is shared - preserve it
                final_env_inds = [self.batch_dim] + final_env_inds

        # 3. Contract
        env_tensor = env_tn.contract(output_inds=final_env_inds)
        
        return env_tensor

    def get_environment_batched(self, tn_base: qt.TensorNetwork, target_tag: str, 
                                 input_batches: List[List[qt.Tensor]],
                                 copy: bool = True, sum_over_batch: bool = False, 
                                 sum_over_output: bool = False):
        """
        Calculates the environment for a target tensor over multiple batches.
        
        Args:
            tn_base: Base TensorNetwork (without inputs)
            target_tag: Tag identifying the tensor to remove
            input_batches: List of batches, where each batch is a list of input tensors
            copy: Whether to copy the network before modification
            sum_over_batch: If True, sums over batch dimension
            sum_over_output: If True, also sums over output dimensions
            
        Returns:
            Environment tensor. If sum_over_batch=False, concatenates batches.
            If sum_over_batch=True, sums on-the-fly across batches.
        """
        if sum_over_batch:
            # Sum on-the-fly - don't store all batch environments
            result = None
            for batch_inputs in input_batches:
                # Create full network with this batch of inputs
                tn_with_inputs = tn_base & batch_inputs
                
                # Get environment for this batch
                env = self.get_environment(tn_with_inputs, target_tag, copy=copy,
                                          sum_over_batch=sum_over_batch, 
                                          sum_over_output=sum_over_output)
                
                if result is None:
                    result = env
                else:
                    # Sum using quimb + operator
                    result = result + env
            
            return result
        else:
            # Concatenate batches - need to collect them
            batch_envs = []
            for batch_inputs in input_batches:
                # Create full network with this batch of inputs
                tn_with_inputs = tn_base & batch_inputs
                
                # Get environment for this batch
                env = self.get_environment(tn_with_inputs, target_tag, copy=copy,
                                          sum_over_batch=sum_over_batch, 
                                          sum_over_output=sum_over_output)
                batch_envs.append(env)
            
            return self._concat_batch_results(batch_envs)

    def forward_with_target(self, tn: qt.TensorNetwork, inputs: List[qt.Tensor], 
                           y: qt.Tensor, mode: str = 'dot', 
                           sum_over_batch: bool = False):
        """
        Forward pass coupled with target output y.
        
        Args:
            tn: TensorNetwork to contract
            inputs: List of input tensors (single batch)
            y: Target output tensor with indices (batch_dim, output_dims...)
            mode: 'dot' for scalar product, 'squared_error' for (forward - y)^2
            sum_over_batch: If True, sums over batch dimension in the result
            
        Returns:
            Result tensor based on mode:
            - 'dot': scalar product forward · y
            - 'squared_error': (forward - y)^2
        """
        if mode == 'dot':
            # Scalar product: add y to the network and contract
            # y has indices (s, y1, y2, ...), forward will match these
            full_tn = tn & inputs & y
            
            if sum_over_batch:
                # Contract everything (sum over all dims including batch)
                result = full_tn.contract(output_inds=[])
            else:
                # Keep batch dimension
                result = full_tn.contract(output_inds=[self.batch_dim])
            
            return result
            
        elif mode == 'squared_error':
            # First compute forward using quimb
            target_inds = [self.batch_dim] + self.output_dimensions
            forward_result = self._batch_forward(tn, inputs, output_inds=target_inds)
            forward_result.transpose_(*target_inds)
            
            # Compute difference using quimb tensor subtraction
            diff = forward_result - y
            
            # Square it using quimb tensor operations
            squared_diff = diff ** 2
            
            # Now we need to sum over output dimensions
            # Create a network with squared_diff and contract over output dims
            if sum_over_batch:
                # Contract all dimensions (sum over batch and outputs)
                result = squared_diff.contract(output_inds=[])
            else:
                # Contract only output dimensions, keep batch
                result = squared_diff.contract(output_inds=[self.batch_dim])
            
            return result
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'dot' or 'squared_error'")

    def _concat_batch_results(self, batch_results: List):
        """
        Concatenate batch results along the batch dimension.
        Uses the appropriate backend (numpy, torch, jax) for concatenation.
        
        Args:
            batch_results: List of tensors or scalars from each batch
            
        Returns:
            Concatenated result (qt.Tensor or scalar)
        """
        if len(batch_results) == 0:
            raise ValueError("No batch results to concatenate")
        
        # Check if results are scalars
        first_result = batch_results[0]
        if not isinstance(first_result, qt.Tensor):
            # Scalars - just sum them if needed or stack
            # For scalars, concatenation doesn't make sense, return as is
            return batch_results
        
        # Get the backend from the data type
        first_data = first_result.data
        
        # Determine backend and concatenate
        if hasattr(first_data, '__array__'):  # numpy-like
            import numpy as np
            concat_data = np.concatenate([t.data for t in batch_results], axis=0)
        else:
            # Try generic concatenate
            try:
                import numpy as np
                concat_data = np.concatenate([t.data for t in batch_results], axis=0)
            except:
                raise NotImplementedError(f"Concatenation not implemented for backend: {type(first_data)}")
        
        return qt.Tensor(concat_data, inds=first_result.inds)
    
    def _sum_batch_results(self, batch_results: List):
        """
        Sum batch results using quimb addition.
        
        Args:
            batch_results: List of tensors or scalars from each batch
            
        Returns:
            Summed result (qt.Tensor or scalar)
        """
        if len(batch_results) == 0:
            raise ValueError("No batch results to sum")
        
        if len(batch_results) == 1:
            return batch_results[0]
        
        # Start with first result
        result = batch_results[0]
        
        # Check if scalar
        if not isinstance(result, qt.Tensor):
            # Scalar - just use + operator
            for t in batch_results[1:]:
                result = result + t
            return result
        
        # Tensor - use quimb addition (element-wise)
        result = result.copy()
        for t in batch_results[1:]:
            result = result + t
        
        return result
