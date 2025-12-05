# type: ignore
from typing import List, Dict, Optional, Tuple
import quimb.tensor as qt
import numpy as np

from tensor.builder import BTNBuilder

# Tag for nodes that should not be trained
NOT_TRAINABLE_TAG = "NT"

class BTN:
    def __init__(self,
                 mu: qt.TensorNetwork,
                 output_dimensions: List[str],
                 batch_dim: str = "s",
                 input_indices: List[str] = None,
                 not_trainable_nodes: List[str] = None
                 ):
        """
        Bayesian Tensor Network.
        
        Args:
            mu: The mean TensorNetwork topology.
            output_dimensions: List of output indices.
            batch_dim: Batch dimension label.
            input_indices: List of input indices (leaf nodes). If None, automatically detected.
            not_trainable_nodes: List of node tags that should not be trained (e.g., input nodes).
                                These nodes will be tagged with NOT_TRAINABLE_TAG ('NT').
        """
        self.mu = mu
        self.output_dimensions = output_dimensions
        self.batch_dim = batch_dim
        
        # Tag not trainable nodes with NT tag
        not_trainable_nodes = not_trainable_nodes or []
        for node_tag in not_trainable_nodes:
            # Get tensor(s) with this tag and add NT tag
            tensors = self.mu.select_tensors(node_tag, which='any')
            for tensor in tensors:
                tensor.add_tag(NOT_TRAINABLE_TAG)
        
        # Store the list of not trainable node tags
        self.not_trainable_nodes = not_trainable_nodes
        
        # Determine input indices
        if input_indices is not None:
            self.input_indices = sorted(input_indices)
        else:
            # Auto-detect: input indices are leaf indices (appear only once)
            ind_count = {}
            for tensor in self.mu:
                for ind in tensor.inds:
                    if ind not in self.output_dimensions and ind != self.batch_dim:
                        ind_count[ind] = ind_count.get(ind, 0) + 1
            self.input_indices = sorted([ind for ind, count in ind_count.items() if count == 1])

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
            

    def prepare_inputs_for_sigma(self, input_data: Dict[str, any], for_sigma: bool = False) -> List[qt.Tensor]:
        """
        Prepare inputs for the sigma network.
        
        For sigma, we need inputs for both regular and primed indices:
        - Regular: same as mu inputs
        - Primed: same data but with _prime suffix on indices
        
        Args:
            input_data: Dictionary mapping input index names to data arrays
            for_sigma: If True, returns inputs for both regular and primed indices
        
        Returns:
            List of input tensors (regular + primed if for_sigma=True)
        """
        # Get regular inputs
        regular_inputs = self.prepare_inputs(input_data)
        
        if not for_sigma:
            return regular_inputs
        
        # Create primed versions
        primed_inputs = []
        for inp in regular_inputs:
            # Prime all indices except batch_dim
            primed_inp = self.prime_indices_tensor(inp, exclude_indices=[self.batch_dim])
            primed_inputs.append(primed_inp)
        
        # Return both regular and primed
        return regular_inputs + primed_inputs

    def _batch_environment(self, tn: qt.TensorNetwork, target_tag: str, copy: bool = True,
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

        # Special case: batch dimension might not be in outer_inds if it's shared
        # between multiple tensors (appears in more than one tensor in the environment).
        # This happens when input tensors all share the batch dimension.
        # In this case, we need to explicitly add it to preserve it.
        if not sum_over_batch and self.batch_dim not in final_env_inds:
            # Check if batch dim exists in the environment
            all_inds_in_env = set()
            for tensor in env_tn:
                all_inds_in_env.update(tensor.inds)
            
            if self.batch_dim in all_inds_in_env:
                # Batch dim exists but is shared (not outer) - we need to preserve it
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
                env = self._batch_environment(tn_with_inputs, target_tag, copy=copy,
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
                env = self._batch_environment(tn_with_inputs, target_tag, copy=copy,
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

    def _sum_over_batches(self, 
                         batch_operation,
                         inputs: any,
                         *args, 
                         **kwargs) -> qt.Tensor:
        """
        Generic method to apply a batch operation and sum results over all batches.
        
        This avoids storing all batch results in memory - sums on-the-fly.
        
        Args:
            batch_operation: Function that takes (batch_idx, inputs, *args, **kwargs)
                           and returns a tensor for that batch
            inputs: Input data prepared with prepare_inputs (list of tensors)
            *args, **kwargs: Additional arguments to pass to batch_operation
        
        Returns:
            Sum of all batch results
            
        Example:
            def my_batch_op(batch_idx, inputs):
                return self._batch_environment(..., batch_idx, ...)
            
            result = self._sum_over_batches(my_batch_op, inputs)
        """
        result = None
        
        # inputs is a list of tensors, get batch size from first one
        num_batches = inputs[0].shape[0]
        
        # Iterate over batches
        for batch_idx in range(num_batches):
            # Compute for this batch
            batch_result = batch_operation(batch_idx, inputs, *args, **kwargs)
            
            # Sum into accumulator
            if result is None:
                result = batch_result
            else:
                result = result + batch_result
        
        return result

    def theta_block_computation(self, 
                                node_tag: str,
                                exclude_bonds: Optional[List[str]] = None) -> qt.Tensor:
        """
        Compute theta^B(i) for a given node: the outer product of expected bond probabilities.
        
        From theoretical model:
            θ^B(i) = ⊗_{b ∈ B(i)} E[λ_b]  where E[λ] = α/β
        
        This creates a tensor representing the expectation of the bond variables (precisions)
        connected to a specific node, excluding output dimensions and optionally other bonds.
        
        Args:
            node_tag: Tag identifying the node in the tensor network
            exclude_bonds: Optional list of bond indices (labels) to exclude from computation.
                          Output dimensions and batch_dim are always excluded.
        
        Returns:
            quimb.Tensor with shape matching the node's shape minus excluded dimensions.
            Acts as a diagonal matrix when used in linear algebra operations.
            
        Example:
            # Node has indices ['a', 'b', 'c', 'out'] with shapes [2, 3, 4, 5]
            # Output dimensions = ['out'], batch_dim = 's'
            # theta = btn.theta_block_computation('node1')
            # Result has indices ['a', 'b', 'c'] with shapes [2, 3, 4]
            # Data is outer product: E[λ_a] ⊗ E[λ_b] ⊗ E[λ_c]
        """
        exclude_bonds = exclude_bonds or []
        
        # Get the node tensor from mu network
        node = self.mu[node_tag]
        excluded_indices = self.output_dimensions + [self.batch_dim] + exclude_bonds 
        
        # Identify bond indices: all indices except output dims, batch dim, and excluded
        bond_indices = [
            ind for ind in node.inds 
            if ind not in excluded_indices 
        ]
        
        if len(bond_indices) == 0:
            raise ValueError(f"Node {node_tag} has no bond indices after exclusions")
        
        # Get expected values E[λ] = α/β for each bond from q_bonds (posterior)
        # These are quimb.Tensor objects with their respective indices
        bond_means = [self.q_bonds[bond_ind].mean() for bond_ind in bond_indices]
        
        # Create TensorNetwork directly from list of tensors
        theta_tn = qt.TensorNetwork(bond_means)
        
        # Contract to get the outer product (preserves all indices and labels)
        theta = theta_tn.contract()
        
        return theta

    def count_trainable_nodes_on_bond(self, bond_ind: str) -> int:
        """
        Count how many trainable nodes share a given bond.
        
        A trainable node is one that does NOT have the NOT_TRAINABLE_TAG ('NT').
        
        Args:
            bond_ind: The bond index (label) to check
        
        Returns:
            Number of trainable nodes that have this bond index
            
        Example:
            # Bond 'a' is shared by node1 (trainable), node2 (trainable), and input (not trainable)
            # count_trainable_nodes_on_bond('a') returns 2
        """
        # Get tensor IDs that have this bond index using ind_map
        tids_with_bond = self.mu.ind_map.get(bond_ind, set())
        
        # Count those that are trainable (don't have NT tag)
        trainable_count = sum(
            1 for tid in tids_with_bond 
            if NOT_TRAINABLE_TAG not in self.mu.tensor_map[tid].tags
        )
        
        return trainable_count

    def prime_indices(self, 
                      node_tag: str, 
                      exclude_indices: Optional[List[str]] = None,
                      prime_suffix: str = "_prime") -> qt.Tensor:
        """
        Create a copy of a node tensor with all indices (except excluded ones) relabeled 
        by appending a suffix.
        
        This is useful for creating 'primed' versions of tensors for operations like
        computing overlaps or conjugate networks.
        
        Args:
            node_tag: Tag identifying the node in the tensor network
            exclude_indices: List of index labels to NOT relabel. 
                           If None, no indices are excluded.
            prime_suffix: Suffix to append to relabeled indices (default: "_prime")
        
        Returns:
            New quimb.Tensor with relabeled indices (not inplace)
            
        Example:
            # Node has indices ('a', 'b', 'c', 'out')
            # prime_indices('node1', exclude_indices=['out'])
            # Returns tensor with indices ('a_prime', 'b_prime', 'c_prime', 'out')
        """
        exclude_indices = exclude_indices or []
        
        # Get the node tensor
        node = self.mu[node_tag]
        
        # Build reindex map: {old_ind: new_ind} for non-excluded indices
        reindex_map = {
            ind: f"{ind}{prime_suffix}"
            for ind in node.inds
            if ind not in exclude_indices
        }
        
        # Reindex returns a new tensor (inplace=False by default)
        primed_node = node.reindex(reindex_map)
        
        return primed_node

    def compute_precision_node(self,
                                   node_tag: str,
                                   inputs: any) -> qt.Tensor:

        tau_expectation = None
        sigma_env = None
        mu_outer_env = None
        theta = None
        original_inds = theta.inds # Get a copy of indices to iterate over

        for old_ind in original_inds:
            primed_ind = old_ind + '_prime'

            # Expand the original index into the desired diagonal pair
            # The original values are placed along the diagonal of (old_ind, primed_ind)
            theta = theta.new_ind_pair_diag(old_ind, old_ind, primed_ind)
        precision = tau_expectation * (sigma_env + mu_outer_env) + theta
        return

    def compute_environment_outer(self,
                                   node_tag: str,
                                   inputs: any) -> qt.Tensor:
        """
        Compute the outer product of mu environment with itself, summed over batches:
        Σ_n (T_i μ x_n) ⊗ (T_i μ x_n)
        
        This is used in computing the precision update for variational inference.
        The environment is computed batch-by-batch, outer product computed, then summed.
        
        From theoretical model (Step 1, node update):
            Part of Σ⁻¹ computation requires: Σ_n (T_i μ x_n) ⊗ (T_i μ x_n)
        
        Args:
            node_tag: Tag identifying the node to exclude from environment
            inputs: Input data prepared with prepare_inputs (list of tensors with batch dim)
        
        Returns:
            Tensor representing Σ_n (T_i μ x_n) ⊗ (T_i μ x_n), with indices being
            the bonds of the node and their primed versions.
            Shape: (bond_a, bond_b, ..., bond_a_prime, bond_b_prime, ...)
        """
        def batch_outer_operation(batch_idx, inputs):
            # Contract mu network with ALL inputs (they have batch dimension)
            tn_with_inputs = self.mu & inputs
            
            # Get environment for this batch (using the tag from mu network)
            env = self._batch_environment(
                tn_with_inputs,
                node_tag,
                sum_over_batch=False,  # Keep batch dim, will select batch_idx
                sum_over_output=False  # Keep output dim for now
            )
            
            # Select the specific batch (env has batch_dim 's')
            # Extract just this batch from the environment
            env_single = env.isel({self.batch_dim: batch_idx})
            
            # Prime indices (exclude output)
            env_prime = self.prime_indices_tensor(env_single, exclude_indices=self.output_dimensions)
            
            # Outer product via tensor network (sums over shared output indices)
            outer_tn = env_single & env_prime
            batch_result = outer_tn.contract()
            
            return batch_result
        
        # Use generic batch summing
        return self._sum_over_batches(batch_outer_operation, inputs)
    
    def prime_indices_tensor(self, 
                            tensor: qt.Tensor,
                            exclude_indices: Optional[List[str]] = None,
                            prime_suffix: str = "_prime") -> qt.Tensor:
        """
        Helper to prime indices of any tensor (not just from network).
        
        Args:
            tensor: The tensor to prime
            exclude_indices: Indices to NOT prime
            prime_suffix: Suffix to add (default "_prime")
        
        Returns:
            New tensor with primed indices
        """
        exclude_indices = exclude_indices or []
        
        reindex_map = {
            ind: f"{ind}{prime_suffix}"
            for ind in tensor.inds
            if ind not in exclude_indices
        }
        
        return tensor.reindex(reindex_map)
  
