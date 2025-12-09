# type: ignore
from torch.distributions import kl_divergence
import importlib
from typing import List, Dict, Optional, Tuple
from jax._src.ops.scatter import Index
import quimb.tensor as qt
import numpy as np
import torch
from tensor.builder import BTNBuilder, Inputs
from tensor.distributions import GammaDistribution
torch.set_default_dtype(torch.float32)   # or torch.float64

# Tag for nodes that should not be trained
NOT_TRAINABLE_TAG = "NT"

class BTN:
    def __init__(self,
                 mu: qt.TensorNetwork,
                 data_stream: Inputs,
                 batch_dim: str = "s",
                 not_trainable_nodes: List[str] = None,
                 method = "cholesky"
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
        self.method = method
        self.mse = None
        self.sigma_forward = None
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
        self.backend = self.mu.backend
        self.p_tau = GammaDistribution(concentration = 1, rate = 1, backend=self.backend)
        self.q_tau = GammaDistribution(concentration = 1, rate = 1, backend=self.backend)
        (self.p_bonds, 
         self.p_nodes, 
         self.q_bonds, 
         self.q_nodes, 
         self.sigma) = builder.build_model()

    def compute_bond_kl(self):
        sum = 0
        for i in self.p_bonds:
            p = self.p_bonds[i].forward()
            q = self.q_bonds[i].forward()
            kl = kl_divergence(q, p).sum()
            print(f"Bond {i}")
            print(kl)
            sum += kl
        return sum

    def compute_node_kl(self):
        sum = 0
        for i in self.p_nodes:
            idx = self.mu[i].inds
            means = [self.p_bonds[i].mean() for i in idx if i not in self.output_dimensions]

            tn = qt.TensorNetwork(means)
            covariance = tn.contract(output_inds = idx)
            p = self.p_nodes[i].forward(cov= torch.diag(covariance.data.view(-1)))
            mu = self.mu[i]
            sigma = self.sigma[i]
            q = self.q_nodes[i].forward(loc = self.mu[i].data, cov = self.sigma[i].data)
            kl = kl_divergence(q, p).sum()
            print(f"Nodes {i}, idx {idx}")
            print(kl)
            sum += kl
        return sum

    def compute_kl(self):
        tau_kl = kl_divergence(self.q_tau.forward(), self.p_tau.forward())
        print("tau kl")
        print(tau_kl)
        return

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
                            sum_over_batch: bool = False,
                            output_inds=[]
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
                sum_over_batch = sum_over_batch,
                output_inds = output_inds
            )
        else:
            result = self._concat_over_batches(
                self._batch_forward_with_target,
                input_generator,
                tn = tn,
                mode = mode,
                sum_over_batch = sum_over_batch,
                output_inds = output_inds
            )
        return result
    
    def _batch_forward_with_target(self,
                                   inputs: List[qt.Tensor], 
                                   y: qt.Tensor,
                                   tn: qt.TensorNetwork,
                                   mode: str = 'dot',
                                   sum_over_batch: bool = False,
                                   output_inds: List[str] = None,
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
        output_inds = [] if output_inds is None else output_inds
        if mode == 'dot':
            # Scalar product: add y to the network and contract
            # y has indices (s, y1, y2, ...), forward will match these
            full_tn = tn & inputs & y
            
            if sum_over_batch:
                # Contract everything (sum over all dims including batch)
                result = full_tn.contract(output_inds=output_inds)
            else:
                # Keep batch dimension
                result = full_tn.contract(output_inds=[self.batch_dim]+output_inds)
            
            return result
            
        elif mode == 'squared_error':
            # First compute forward using quimb
            target_inds = [self.batch_dim] + self.output_dimensions
            forward_result = self._batch_forward(inputs, tn, output_inds=target_inds)
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

    def _get_concentration_update(self, bond_tag):
        bond_concentration = self.p_bonds[bond_tag].concentration
        num_trainable_nodes, _ = self.count_trainable_nodes_on_bond(bond_tag)
        bond_dim = self.mu.inds_size([bond_tag])
        update = bond_concentration +  bond_dim * num_trainable_nodes * 0.5
        return update

    def _get_rate_update(self, bond_tag):
        p_rate = self.p_bonds[bond_tag].rate
        _, tag_trainable_nodes = self.count_trainable_nodes_on_bond(bond_tag)
        update = None
        for i in tag_trainable_nodes:
            mu_node = self.mu[i]**2
            sigma_node = self._unprime_indices_tensor(self.sigma[i])
            partial_mu = self._get_partial_trace(mu_node, i, bond_tag)
            partial_sigma = self._get_partial_trace(sigma_node, i, bond_tag)
            if update is None:
                update= 0.5 * (partial_mu + partial_sigma)
            else:
                update += 0.5* (partial_mu + partial_sigma)
        return p_rate + update

    def update_bond(self, bond_tag):
        concentration_update = self._get_concentration_update(bond_tag)
        rate_update = self._get_rate_update(bond_tag)
        self.q_bonds[bond_tag].update_parameters(
                        concentration = concentration_update,
                        rate = rate_update
                    )
        return

    def _calc_mu_mse(self, inputs= None):
        if inputs is None:
            inputs = self.data
        mu_mse = self.forward_with_target(
                    inputs.data_mu_y,
                    self.mu,
                    'squared_error',
                    True,
                    []
                )
        self.mse = mu_mse
        return mu_mse

    def _calc_sigma_forward(self, inputs = None):
        if inputs is None:
            inputs = self.data
        sigma_forward = self.forward(self.sigma, inputs.data_sigma, True, True)
        return sigma_forward
    
    def _get_tau_update(self):
        concentration = self.p_tau.concentration + self.data.samples * 0.5
        mse = self._calc_mu_mse()
        sigma_f = self._calc_sigma_forward() 
        rate = self.p_tau.rate + 0.5 * (mse + sigma_f)
        return concentration, rate

    def update_tau(self):
        concentration, rate = self._get_tau_update()
        self.q_tau.update_parameters(
                            concentration = concentration,
                            rate = rate
                         )

    def _get_partial_trace(self, node: qt.Tensor, node_tag, bond_tag):
        theta = self.theta_block_computation(node_tag, [bond_tag])
        tn = node & theta
        return tn.contract(output_inds=[bond_tag])

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
        tids = self.mu.ind_map.get(bond_ind, set())
        trainable = [
            self.mu.tensor_map[tid]
            for tid in tids
            if NOT_TRAINABLE_TAG not in self.mu.tensor_map[tid].tags
        ]
        return len(trainable), [t.tags for t in trainable]

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

    def _get_mu_update(self, node_tag, debug = False):
        mu_idx = self.mu[node_tag].inds
        self.mu.delete(node_tag)
        rhs = self.forward_with_target(
            self.data.data_mu_y,
            self.mu,
            'dot',
            sum_over_batch = True,
            output_inds = mu_idx
        )
        relabel_map = {}
        for ind in mu_idx:
            if ind in self.output_dimensions:
                pass 
            else:
                relabel_map[ind] = ind + '_prime'
        rhs = rhs.reindex(relabel_map)
        rhs = rhs * self.q_tau.mean()
        tn = rhs & self.sigma[node_tag]

        mu_update = tn.contract(output_inds=mu_idx)
        mu_update.modify(tags=[node_tag])
        if debug:
            return mu_update, rhs, self.sigma[node_tag]
        return mu_update
    
    def update_sigma_node(self, node_tag):
        sigma_update = self._get_sigma_update(node_tag)
        self.update_node(self.sigma, sigma_update, node_tag)
        return

    def update_mu_node(self, node_tag):
        mu_update = self._get_mu_update(node_tag)
        self.update_node(self.mu, mu_update, node_tag)
        return
    
    def update_node(self, tn, tensor, node_tag):
        """
        Updates the tensor network by replacing the node with the given tag
        with the new tensor.
        """

        if node_tag in tn.tag_map:
            tn.delete(node_tag)
        
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
        Inverts a tensor treating 'index_bases' as the column indices 
        and 'index_bases + _prime' as the row indices.
        """
        if tag is None:
            tag = tensor.tags
            
        # 1. Sort indices for consistent ordering (Block Layout)
        # col_inds = Unprimes (e.g., 'b1', 'x2')
        # row_inds = Primes   (e.g., 'b1_prime', 'x2_prime')
        col_inds = sorted(index_bases)
        row_inds = [i + '_prime' for i in col_inds]

        # 2. Extract Matrix: (Rows=Unprimes, Cols=Primes)
        # We explicitly map Unprimes to Rows to match standard linear algebra 
        # A * x = b  (A has rows matching x's indices)
        matrix_data = tensor.to_dense(col_inds, row_inds)

        # 3. Detect Backend & Invert
        backend_name, lib = self.get_backend(matrix_data)
        
        if method == 'cholesky':
            inv_data = self.cholesky_invert(matrix_data, backend_name, lib)
        else:
            if backend_name in ('torch', 'jax', 'numpy'):
                inv_data = lib.linalg.inv(matrix_data)
            else:
                raise ValueError(f"Unknown backend '{backend_name}' for inversion.")

        # 4. Reshape & Return
        # The matrix 'inv_data' is (Unprimes, Primes). 
        # We must assign indices in that exact order.
        new_shape = tuple(tensor.ind_size(i) for i in col_inds + row_inds)
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
        env_prime = self._prime_indices_tensor(env, exclude_indices=self.output_dimensions+[self.batch_dim])
        # Outer product via tensor network (sums over shared output indices)

        env_inds = env.inds + env_prime.inds
        outer_tn = env & env_prime
        out_indices = sample_dim + [i for i in env_inds if i not in [self.batch_dim]]
        batch_result = outer_tn.contract(output_inds = out_indices)
        return batch_result
        
    
    def _prime_indices_tensor(self, 
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
  
    def _unprime_indices_tensor(self, tensor: qt.Tensor, prime_suffix: str = "_prime") -> qt.Tensor:

        reindex_map = {
            ind: ind[:-len(prime_suffix)]
            for ind in tensor.inds
            if ind.endswith(prime_suffix)
        }
        return tensor.reindex(reindex_map)

    def fit(self, epochs):
        bonds = [i for i in self.mu.ind_map if i not in self.output_dimensions]
        nodes = list(self.mu.tag_map.keys())
        for i in range(epochs):
            # self.debug_print()
            for node_tag in nodes:
                self.update_sigma_node(node_tag)
                self.update_mu_node(node_tag)
            for bond_tag in bonds:
                self.update_bond(bond_tag)
            self.update_tau()
            ctau = self.q_tau.concentration
            rtau = self.q_tau.rate
            mse = self._calc_mu_mse()/self.data.samples
            print(f"MSE {mse.data:.4f}, E[t] {self.get_tau_mean()}, C = {ctau}, R = {rtau:.4f}")
        return

    def debug_print(self, node_tag=None, bond_tag=None):
        bonds = [i for i in self.mu.ind_map if i not in self.output_dimensions] if bond_tag is None else [bond_tag]
        nodes = list(self.mu.tag_map.keys()) if node_tag is None else [node_tag]

        print("="*30)
        print(" CONCENTRATIONS ".center(30, "="))
    
        for b in bonds:
            c = self.q_bonds[b].concentration
            r = self.q_bonds[b].rate
            print(f"{b}: conc={np.array2string(c.data, precision=3, separator=',')}", 
                  f"rate={np.array2string(r.data, precision=3, separator=',')}")

        print("="*30)
        print(" MU & SIGMA SUMS ".center(30, "="))
    
        for n in nodes:
            mu_val = self.mu[n].data.sum()
            sigma_val = self.sigma[n].data.sum()
            print(f"{n}: mu={mu_val:.3f}, sigma={sigma_val:.3f}")

        print("="*30)

