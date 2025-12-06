# type: ignore
import importlib

from logging import ERROR
from typing import List, Dict, Optional, Tuple
import quimb.tensor as qt
import numpy as np

from tensor.builder import BTNBuilder, Inputs
from tensor.distributions import GammaDistribution

# Tag for nodes that should not be trained
NOT_TRAINABLE_TAG = "NT"

class BTN:
    def __init__(self,
                 mu: qt.TensorNetwork,
                 data_stream: Inputs,
                 batch_dim: str = "s",
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
        self.output_dimensions = data_stream.outputs_labels
        self.batch_dim = batch_dim
        self.input_indices = data_stream.input_labels
        self.data = data_stream
        # Tag not trainable nodes with NT tag
        not_trainable_nodes = not_trainable_nodes or []
        for node_tag in not_trainable_nodes:
            # Get tensor(s) with this tag and add NT tag
            tensors = self.mu.select_tensors(node_tag, which='any')
            for tensor in tensors:
                tensor.add_tag(NOT_TRAINABLE_TAG)
        # Store the list of not trainable node tags
        self.not_trainable_nodes = not_trainable_nodes
      
        # --- Initialize Builder and Model Components ---
        builder = BTNBuilder(self.mu, self.output_dimensions, self.batch_dim)
        
        # Build and unpack components
        # p_bonds: Priors for edges (Gamma)
        # p_nodes: Priors for nodes (Gaussian)
        # q_bonds: Posteriors for edges (Gamma)
        # q_nodes: Posteriors for nodes (Gaussian)
        # sigma: The covariance TensorNetwork
        self.p_tau = GammaDistribution(concentration = 1, rate = 1)
        self.q_tau = GammaDistribution(concentration = 1, rate = 1)
        (self.p_bonds, 
         self.p_nodes, 
         self.q_bonds, 
         self.q_nodes, 
         self.sigma) = builder.build_model()

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

    def _batch_forward(self, inputs: List[qt.Tensor], tn, output_inds: List[str]) -> qt.Tensor:
        """Helper for forward pass, contracting a single batch of inputs."""

        full_tn = tn & inputs
        res = full_tn.contract(output_inds=output_inds)
        if len(output_inds) > 0:
            res.transpose_(*output_inds)
        return res 

    def forward(self, tn: qt.TensorNetwork, input_generator, 
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
            result = self._sum_over_batches(
                self._batch_forward,
                input_generator,
                tn = tn,
                output_inds = target_inds
            )
        else:
            result = self._concat_over_batches(
                self._batch_forward,
                input_generator,
                tn = tn,
                output_inds = target_inds
            )
        return result
    
    def get_environment(self, tn: qt.TensorNetwork,
                                 target_tag: str, 
                                 input_generator,
                                 copy: bool = True,
                                 sum_over_batch: bool = False, 
                                 sum_over_output: bool = False
                             ):
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
        if copy:
            tn_base = tn.copy()
        else:
            tn_base = tn
        if sum_over_batch:
            result = self._sum_over_batches(
                                self._batch_environment,
                                input_generator,
                                tn = tn_base,
                                target_tag=target_tag,
                                sum_over_batch=sum_over_batch,
                                sum_over_output=sum_over_output
                            )
        else:
            result = self._concat_over_batches(
                                self._batch_environment,
                                input_generator,
                                tn = tn_base,
                                target_tag=target_tag,
                                sum_over_batch=sum_over_batch,
                                sum_over_output=sum_over_output
                            )
        return result

    def _batch_environment(self, inputs, tn: qt.TensorNetwork, target_tag: str,
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
        env_tn= tn & inputs
        # 1. Create the "hole"
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

    def forward_with_target(
                            self,
                            input_generator,
                            tn: qt.TensorNetwork,
                            mode: str = 'dot',
                            sum_over_batch: bool = False
                        ):
        """
           Forward to pass a dot product or compute MSE.
           The generator should return mu_y or sigma_y
        """
        if sum_over_batch:
            result = self._sum_over_batches(
                self._batch_forward_with_target,
                input_generator,
                tn = tn,
                mode = mode,
                sum_over_batch = sum_over_batch
            )
        else:
            result = self._concat_over_batches(
                self._batch_forward_with_target,
                input_generator,
                tn = tn,
                mode = mode,
                sum_over_batch = sum_over_batch
            )
        return result
    
    def _batch_forward_with_target(self,
                                   inputs: List[qt.Tensor], 
                                   y: qt.Tensor,
                                   tn: qt.TensorNetwork,
                                   mode: str = 'dot',
                                   sum_over_batch: bool = False
                                ):
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

    def _concat_over_batches(self, 
                             batch_operation, 
                             data_iterator, 
                             *args, 
                             **kwargs
                         ):
        """
        Iterates over data_iterator, collects results, and concatenates them.
        """
        results = []
        
        for batch_idx, batch_data in enumerate(data_iterator):
            # Ensure proper unpacking (handle tuple vs single item)
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)
            
            # Execute operation
            batch_res = batch_operation(*inputs, *args, **kwargs)
            results.append(batch_res)
            
        return self._concat_batch_results(results)

    def _sum_over_batches(self, 
                          batch_operation, 
                          data_iterator, 
                          *args, 
                          **kwargs) -> qt.Tensor:
        """
        Args:
            batch_operation: Function accepting (batch_idx, *unpacked_data, *args, **kwargs)
            data_iterator: The generator property (e.g., loader.mu_sigma_y_batches)
        """
        result = None
        
        for batch_data in data_iterator:
            # Ensure data is a tuple for unpacking
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)
            
            # Unpack inputs into the operation (e.g., mu, sigma, y)
            batch_result = batch_operation(*inputs, *args, **kwargs)
            
            result = batch_result if result is None else result + batch_result
            
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

    def get_tau_mean(self):
        return self.q_tau.mean()

    def compute_precision_node(self,
                                   node_tag: str,
                               ) -> qt.Tensor:

        tau_expectation = self.get_tau_mean()
        sigma_env = self.get_environment(
                         tn =self.sigma,
                         target_tag=node_tag ,
                         input_generator=self.data.data_sigma,
                         copy=False,
                         sum_over_batch = True,
                         sum_over_output=True
                     )

        mu_outer_env = self.outer_operation(
            tn=self.mu,
            node_tag=node_tag,
            input_generator=self.data.data_mu,
            sum_over_batches=True
        )

        theta = self.theta_block_computation(
            node_tag=node_tag,
        )
        original_inds = theta.inds # Get a copy of indices to iterate over

        for old_ind in original_inds:
            primed_ind = old_ind + '_prime'
            # Expand the original index into the desired diagonal pair
            # The original values are placed along the diagonal of (old_ind, primed_ind)
            theta = theta.new_ind_pair_diag(old_ind, old_ind, primed_ind)
        precision = tau_expectation * (sigma_env + mu_outer_env) + theta
        # Broadcast over original output dimension
        # To do after inverting.
        # The inverse of a broadcast is the broadcast of the inverse o.o
        # Just because we only care of the diagonal part of the broadcasted dimension, so It is the same as outer producting with an identity
        return precision

    def _get_sigma_update(self, node_tag):
        block_output_ind = [i for i in self.mu[node_tag].inds if i in self.output_dimensions]
        block_variational_ind = [i for i in self.mu[node_tag].inds if i not in self.output_dimensions]
        precision = self.compute_precision_node(node_tag)
        tag = node_tag 
        sigma_node = self.invert_ordered_tensor(precision, block_variational_ind, tag = tag)
        for i in block_output_ind:
            sigma_node.new_ind(i, self.mu.ind_size(i))
        return sigma_node

    def update_sigma_node(self, node_tag):
        sigma_update = self._get_sigma_update(node_tag)
        # TODO: + _sigma? or lets call all nodes with same tag de
        self.update_node(self.sigma, sigma_update, node_tag)
        return
    
    def update_node(self, tn, tensor, node_tag):
        """
        Updates the tensor network by replacing the node with the given tag
        with the new tensor.
        """
        # Identify tensors to remove (collect first to avoid iterator modification issues)
        to_remove = [t for t in tn if node_tag in t.tags]
    
        # Remove existing
        for t in to_remove:
            tn.remove_tensor(t)
        
        # Add new
        tn.add_tensor(tensor)
    
    def get_backend(self, data):
        module = type(data).__module__
        if 'torch' in module:
            return 'torch', importlib.import_module('torch')
        elif 'jax' in module:
            return 'jax', importlib.import_module('jax.numpy')
        elif 'numpy':
            return 'numpy', np

    def cholesky_invert(self, matrix, backend_name, lib):
        """Inverts Hermitian positive-definite matrix via Cholesky."""
        if backend_name == 'torch':
            # Torch has a dedicated cholesky_inverse
            L = lib.linalg.cholesky(matrix)
            return lib.cholesky_inverse(L)
        elif backend_name == 'jax':
            # JAX requires solve against identity or custom implementation
            L = lib.linalg.cholesky(matrix)
            id_mat = lib.eye(matrix.shape[0], dtype=matrix.dtype)
            return lib.linalg.solve(matrix, id_mat) # Often more stable than explicit inv
        elif backend_name == 'numpy':
            try:
                L = lib.linalg.cholesky(matrix)
                inv_L = lib.linalg.inv(L)
                return inv_L.T.conj() @ inv_L
            except lib.linalg.LinAlgError:
                return lib.linalg.inv(matrix)
        else:
            raise ValueError(f"Unknown backend '{backend_name}' for inversion.")

    def invert_ordered_tensor(self, tensor, index_bases, method='cholesky', tag=None):
        """
        Args:
            tensor: quimb.Tensor or TensorNetwork
            index_bases: list of str (the base index names, e.g. ['k', 'a'])
            method: 'direct' or 'cholesky'
        """
        if tag is None:
            tag = tensor.tags
        # 1. Sort indices for consistent ordering
        # We define rows as the base indices, cols as base + '_prime'
        row_inds = sorted(index_bases)
        col_inds = [i + '_prime' for i in row_inds]
     # 2. Transpose Tensor to (Row_Inds, Col_Inds)
        # This orders the data contiguously in memory for the matrix view
        tensor = tensor.transpose(*row_inds, *col_inds)
        # 2. Extract Matrix
        # to_dense(rows, cols) handles the random internal ordering automatically
        matrix_data = tensor.to_dense(row_inds, col_inds)

        # 3. Detect Backend
        backend_name, lib = self.get_backend(matrix_data)

        # 4. Invert
        if method == 'cholesky':
            inv_data = self.cholesky_invert(matrix_data, backend_name, lib)
        else:
            # Standard inversion
            if backend_name in ('torch', 'jax', 'numpy'):
                inv_data = lib.linalg.inv(matrix_data)
            else:
                raise ValueError(f"Unknown backend '{backend_name}' for inversion.")

        sizes = tensor.ind_sizes()
        new_shape = tuple(sizes[i] for i in col_inds + row_inds)
    
        inv_tensor_data = inv_data.reshape(new_shape)

        return qt.Tensor(data=inv_tensor_data, inds=col_inds + row_inds, tags=tag)        

    def outer_operation(self, input_generator, tn, node_tag, sum_over_batches):
        if sum_over_batches:
            result = self._sum_over_batches(
                self._batch_outer_operation,
                input_generator,
                tn = tn,
                node_tag = node_tag,
                sum_over_batches=sum_over_batches
            )
        else:
            result = self._concat_over_batches(
                self._batch_outer_operation,
                input_generator,
                tn = tn,
                node_tag = node_tag,
                sum_over_batches=sum_over_batches
            )
        return result
    
    def _batch_outer_operation(self, inputs, tn, node_tag, sum_over_batches: bool):
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

        env = self._batch_environment(
            inputs,
            tn,
            target_tag=node_tag,
            sum_over_batch=False,
            sum_over_output=True
        )
      
        sample_dim = [self.batch_dim] if not sum_over_batches else []
        # Prime indices (exclude output)
        env_prime = self.prime_indices_tensor(env, exclude_indices=self.output_dimensions+[self.batch_dim])
        # Outer product via tensor network (sums over shared output indices)

        env_inds = env.inds + env_prime.inds
        outer_tn = env & env_prime
        out_indices = sample_dim + [i for i in env_inds if i not in [self.batch_dim]]
        batch_result = outer_tn.contract(output_inds = out_indices)
        return batch_result
        
    
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
  
