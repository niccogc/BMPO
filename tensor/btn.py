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
        self.threshold = 0.8
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

    def compute_bond_kl(self, verbose=False):
        """
        Compute KL divergence for all bond (edge) distributions.
        
        Args:
            verbose: If True, print individual bond KL values
            
        Returns:
            Total bond KL divergence
        """
        total_kl = 0
        for i in self.p_bonds:
            p = self.p_bonds[i].forward()
            q = self.q_bonds[i].forward()
            kl = kl_divergence(q, p).sum()
            if verbose:
                print(f"Bond {i}: KL = {kl:.4f}")
            total_kl += kl
        return total_kl

    def compute_node_kl(self, verbose=False, debug=False):
        """
        Compute KL divergence between q (posterior) and p (prior) for all nodes.
        
        Args:
            verbose: If True, print individual node KL values
            debug: If True, print detailed debugging information
        
        KEY CHANGE: Output dimensions are now treated as variational parameters!
        - All indices (including output) have bond priors
        - Prior: p(θ) = N(0, Σ_bonds) where Σ_bonds includes ALL dimensions
        - Posterior: q(θ) has learned mean and covariance for ALL dimensions
        - Just ONE KL per node: KL[q(θ) || p(θ)]
        
        FIXED: theta represents PRECISIONS (E[λ]), so covariance = 1/theta (element-wise)
        """
        total_kl = 0
        for i in self.p_nodes:
            variational_mu, output_mu, mu_idx = self.get_variational_outputs_inds(self.mu[i])
            variational_sigma, _, sigma_idx = self.get_variational_outputs_inds(self.sigma[i])
            mu_shape = self.mu[i].shape
            
            if debug:
                print(f"\n{'='*60}")
                print(f"DEBUG: Node {i}")
                print(f"{'='*60}")
                print(f"mu_idx: {mu_idx}")
                print(f"mu_shape: {mu_shape}")
                print(f"sigma_idx: {sigma_idx}")
            
            if debug:
                print(f"variational_inds (includes output): {variational_mu}")
            
            # Get theta (PRECISION from bond priors) - use prior bonds
            theta = self.theta_block_computation(i, use_prior=True)
            
            # FIXED: Convert precision to covariance BEFORE diagonalization
            # Transpose to mu_idx ordering, then do element-wise 1/theta
            theta = theta.transpose(*mu_idx)
            theta_data = theta.data.flatten()
            
            # Covariance = 1/precision (element-wise)
            cov_data = 1.0 / theta_data
            
            # Create diagonal covariance matrix
            prior_cov = torch.diag(cov_data)
            
            prior_loc = torch.zeros_like(self.mu[i].data.flatten())
            p = self.p_nodes[i].forward(loc=prior_loc, cov=prior_cov)
            
            if debug:
                print(f"Prior p: mean shape = {prior_loc.shape}, cov shape = {prior_cov.shape}")
            
            # Q distribution: posterior q(theta)
            # Flatten mu and sigma
            mu_flat = self.mu[i].data.flatten()
            
            # Transpose sigma to have proper ordering: [var_inds] + [var_inds_prime]
            primed_mu_idx = [inds + '_prime' for inds in mu_idx]
            q_cov = self.sigma[i]
            for out in output_mu:
                primed = out + '_prime'
                q_cov = q_cov.new_ind_pair_diag(out, out, primed)
            q_cov = q_cov.to_dense(mu_idx, primed_mu_idx)
            q = self.q_nodes[i].forward(loc=mu_flat, cov=q_cov)
            
            # Compute KL divergence (just ONE per node now!)
            node_kl = kl_divergence(q, p)
            
            if debug or verbose:
                print(f"\nNode {i}: KL = {node_kl:.4f} (single KL, output dims are variational)")
            
            total_kl += node_kl
            
        return total_kl

    def compute_tau_kl(self):
        return kl_divergence(self.q_tau.forward(), self.p_tau.forward())
   
    def compute_expected_log_likelihood(self, verbose = False):
        """
        Compute E_q[log p(y|θ,τ)] - the expected log likelihood of the data.
        
        For Gaussian likelihood p(y|μ,τ) with precision τ:
        log p(y|μ,τ) = -0.5 * τ * ||y - μx||² + 0.5 * log(τ) - 0.5 * log(2π)
        
        Taking expectation over q(θ,τ):
        E_q[log p(y|θ,τ)] = -0.5 * E[τ] * (MSE + sigma_forward) 
                           + 0.5 * N * E_q[log τ] 
                           - 0.5 * N * log(2π)
        
        Where:
        - MSE = Σ_n (y_n - μ_x_n)² (sum over all data points)
        - sigma_forward = forward(sigma) (variance contribution from σ)
        - N = number of data points × output dimension size
        - E[τ] = concentration/rate (mean of Gamma)
        - E_q[log τ] = digamma(concentration) - log(rate) for Gamma distribution
        
        Returns:
            Scalar value of expected log likelihood
        """
        mse_sigma = self.q_tau.rate - self.p_tau.rate
        if mse_sigma is None:
            # Compute MSE: E_q[(y - μx)²]
            mse = self._calc_mu_mse()
            # Compute sigma forward: E_q[forward(σ)]
            sigma_forward = self._calc_sigma_forward()
            mse_sigma = (mse + sigma_forward)*0.5

        # Get E[τ] (mean of q_tau)
        tau_mean = self.q_tau.mean()
        
        # Get E_q[log τ] for Gamma distribution
        # For Gamma(α, β): E[log X] = digamma(α) - log(β)
        concentration = self.q_tau.concentration
        rate = self.q_tau.rate
        
        # Extract data if they're quimb tensors and convert to torch tensors
        if isinstance(concentration, qt.Tensor):
            concentration = concentration.data
        else:
            concentration = torch.tensor(concentration, dtype=torch.float32)
            
        if isinstance(rate, qt.Tensor):
            rate = rate.data
        else:
            rate = torch.tensor(rate, dtype=torch.float32)
            
        if isinstance(tau_mean, qt.Tensor):
            tau_mean = tau_mean.data
            
        expected_log_tau = torch.digamma(concentration) - torch.log(rate)
        
        # Number of data points (samples × output dimension size)
        # output_dimensions is a list of output labels
        output_dim_size = 1
        for out_label in self.output_dimensions:
            output_dim_size *= self.mu.ind_size(out_label)
        
        N = self.data.samples * output_dim_size
        
        # Compute expected log likelihood
        # E_q[log p(y|θ,τ)] = -0.5 * E[τ] * (MSE + sigma_forward) 
        #                    + 0.5 * N * E_q[log τ] 
        #                    - 0.5 * N * log(2π)
        
        likelihood_term1 = -tau_mean * mse_sigma
        likelihood_term2 = 0.5 * N * expected_log_tau
        likelihood_term3 = -0.5 * N * np.log(2 * np.pi)
        
        expected_log_likelihood = likelihood_term1 + likelihood_term2 + likelihood_term3
        if verbose:    
            print("\n=== Expected Log Likelihood ===")
            print(f"MSE term: {mse}")
            print(f"Sigma forward term: {sigma_forward}")
            print(f"E[τ]: {tau_mean}")
            print(f"E[log τ]: {expected_log_tau}")
            print(f"N (samples × output_dim): {N}")
            print(f"Term 1 (-0.5 * E[τ] * (MSE + sigma_forward)): {likelihood_term1}")
            print(f"Term 2 (0.5 * N * E[log τ]): {likelihood_term2}")
            print(f"Term 3 (-0.5 * N * log(2π)): {likelihood_term3}")
            print(f"Total Expected Log Likelihood: {expected_log_likelihood}")
        
        return expected_log_likelihood
    
    def compute_elbo(self, verbose=True, print_components=False):
        """
        Compute the Evidence Lower BOund (ELBO).
        
        ELBO = E_q[log p(y|θ,τ)] - KL[q(θ)||p(θ)] - KL[q(τ)||p(τ)]
        
        The ELBO should INCREASE during training (become less negative).
        It is a lower bound on log p(y), the log marginal likelihood.
        
        Where:
        - E_q[log p(y|θ,τ)] is the expected log likelihood
        - KL[q(θ)||p(θ)] is the sum of:
            - KL divergence for all bonds (edges)
            - KL divergence for all nodes (blocks)
        - KL[q(τ)||p(τ)] is the KL divergence for the precision parameter
        
        Args:
            verbose: If True, print summary of ELBO computation
            print_components: If True, print detailed breakdown of likelihood
        
        Returns:
            ELBO value (scalar)
        """

        # Compute KL divergences (without printing)
        expected_log_lik = self.compute_expected_log_likelihood(verbose=False)
        bond_kl = self.compute_bond_kl(verbose=print_components)
        node_kl = self.compute_node_kl(verbose=print_components)
        tau_kl = self.compute_tau_kl()
        
        # Extract scalar if it's a tensor
        if isinstance(tau_kl, torch.Tensor):
            tau_kl = tau_kl.item() if tau_kl.numel() == 1 else tau_kl.sum().item()
        
        # ELBO = Expected Log Likelihood - Total KL
        total_kl = bond_kl + node_kl + tau_kl
        elbo = expected_log_lik - total_kl
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ELBO COMPUTATION")
            print(f"{'='*60}")
            print(f"Expected Log Likelihood: {expected_log_lik:.4f}")
            print(f"Bond KL: {bond_kl:.4f}")
            print(f"Node KL: {node_kl:.4f}")
            print(f"Tau KL: {tau_kl:.4f}")
            print(f"Total KL: {total_kl:.4f}")
            print(f"ELBO = E[log p(y|θ,τ)] - KL: {elbo:.4f}")
            print(f"{'='*60}\n")
        
        return elbo

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
                                exclude_bonds: Optional[List[str]] = None,
                                use_prior: bool = False) -> qt.Tensor:
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
            use_prior: If True, use p_bonds (prior). If False, use q_bonds (posterior).
                      Use True for computing prior covariance in KL.
                      Use False for precision updates (default).
        
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
        excluded_indices = [self.batch_dim] + exclude_bonds 
        
        # Identify bond indices: all indices except output dims, batch dim, and excluded
        bond_indices = [
            ind for ind in node.inds 
            if ind not in excluded_indices 
        ]
        
        if len(bond_indices) == 0:
            raise ValueError(f"Node {node_tag} has no bond indices after exclusions")
        
        # Choose which bond distributions to use
        bond_dist = self.p_bonds if use_prior else self.q_bonds
        
        # Get expected values E[λ] = α/β for each bond
        # These are quimb.Tensor objects with their respective indices
        bond_means = [bond_dist[bond_ind].mean() for bond_ind in bond_indices]
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

    def _calc_mu_forward(self, inputs = None):
        if inputs is None:
            inputs = self.data
        mu_forward = self.forward(self.mu, inputs.data_mu, True, True)
        return mu_forward
    
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

    def get_variational_outputs_inds(self,node):
        node_inds = node.inds
        node_output = []
        node_variational = []
        for i in node_inds:
            if i in self.output_dimensions:
                node_output.append(i)
            else:
                node_variational.append(i)

        return node_variational, node_output, node_inds
    
    def compute_precision_node(self,
                                   node_tag: str,
                               ) -> qt.Tensor:

        tau_expectation = self.get_tau_mean()

        variational_inds, output_inds, node_inds = self.get_variational_outputs_inds(self.mu[node_tag])
        # Calculate output dimension size (product of all output dims for this node)
        output_dim_size = 1
        for out_ind in output_inds:
            output_dim_size *= self.mu.ind_size(out_ind)

        # Compute environment terms (these sum over output, so we multiply by output_dim_size)
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
        
        # Compute theta (bond prior contribution) - includes output dimensions
        theta = self.theta_block_computation(
            node_tag=node_tag,
        )

        for old_ind in variational_inds:
            primed_ind = old_ind + '_prime'
            # Expand the original index into the desired diagonal pair
            # The original values are placed along the diagonal of (old_ind, primed_ind)
            theta = theta.new_ind_pair_diag(old_ind, old_ind, primed_ind)

        # Outer product sigma_env and mu_outer_env with identity over output dims to match theta
        # sigma_env and mu_outer_env have structure [variational] + [variational_prime] (no output)
        # theta has structure [all_inds] + [all_inds_prime] (includes output with primes)
        # We need to add output × output_prime dimensions to match
        second_moments = tau_expectation * output_dim_size * (sigma_env + mu_outer_env)

        for out_ind in output_inds:
            out_size = self.mu.ind_size(out_ind)
            second_moments.new_ind(out_ind, out_size, mode='repeat')
        precision = second_moments + theta
        
        return precision

    def _get_sigma_update(self, node_tag):
        # All non-batch indices are now variational (including output!)
        variational_ind, output_ind, node_ind = self.get_variational_outputs_inds(self.mu[node_tag])
        precision = self.compute_precision_node(node_tag)
        variational_ind_prime = [i + '_prime' for i in variational_ind]
        map = {'o' : output_ind, 'rows' : variational_ind, 'cols': variational_ind_prime}
        var_sizes = tuple(self.mu[node_tag].ind_size(i) for i in variational_ind)
        out_sizes = tuple(self.mu[node_tag].ind_size(i) for i in output_ind)

        # 3. Define the Shape Map (mapping fused index -> tuple of new sizes)
        shape_map = {
            'o': out_sizes,
            'rows': var_sizes,
            'cols': var_sizes  # 'cols' uses the same sizes as 'rows'
        }
        precision.fuse(map, inplace=True)
        matrix_data = precision.to_dense(['o'], ['rows'], ['cols'])
        tensor_sigma_node = self.invert_ordered_tensor(matrix_data)
        sigma_node = qt.Tensor(tensor_sigma_node, ["o", 'rows', 'cols'], tags = node_tag)
        sigma_node.unfuse(map, shape_map=shape_map,inplace=True)
        return sigma_node

    def _get_mu_update(self, node_tag, debug = False):
        variational_idx, output_idx, mu_idx = self.get_variational_outputs_inds(self.mu[node_tag])
        self.mu.delete(node_tag)
        rhs = self.forward_with_target(
            self.data.data_mu_y,
            self.mu,
            'dot',
            sum_over_batch = True,
            output_inds = mu_idx
        )
        relabel_map = {}
        for ind in variational_idx:
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

        if backend_name == 'torch':
            L = lib.linalg.cholesky(matrix)            # (B,n,n)
            return lib.cholesky_inverse(L)             # (B,n,n)

        elif backend_name == 'jax':
            L = lib.linalg.cholesky(matrix)            # (B,n,n)
            R = lib.eye(matrix.shape[-1], dtype=matrix.dtype)
            R = lib.broadcast_to(R, matrix.shape)      # (B,n,n)
            return lib.linalg.solve(matrix, R)         # (B,n,n)

        elif backend_name == 'numpy':
            M = matrix
            if M.ndim == 2:   # single matrix
                try:
                    L = lib.linalg.cholesky(M)
                    invL = lib.linalg.inv(L)
                    return invL.T.conj() @ invL
                except lib.linalg.LinAlgError:
                    return lib.linalg.inv(M)

            out = []
            for Mi in M:      # batch case
                try:
                    L = lib.linalg.cholesky(Mi)
                    invL = lib.linalg.inv(L)
                    out.append(invL.T.conj() @ invL)
                except lib.linalg.LinAlgError:
                    out.append(lib.linalg.inv(Mi))
            return lib.array(out)
        else:
            raise ValueError(f"Unknown backend '{backend_name}'.")


    def invert_ordered_tensor(self, matrix_data, method='cholesky'):
        """
            Matrix data can should be batched
        """

        backend_name, lib = self.get_backend(matrix_data)

        if method == 'cholesky':
            inv_data = self.cholesky_invert(matrix_data, backend_name, lib)
        else:
            inv_data = lib.linalg.inv(matrix_data)

        return inv_data


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

   
    def _trim_bond(self, bond_tag, verbose = False):
        weights = self.q_bonds[bond_tag].mean().data
        threshold = self.threshold
        # --- 3. SORT & TRACK INDICES ---
        # argsort(descending=True) returns the INDICES of the values from High -> Low
        # This effectively "keeps track of the position of the original"
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]

        # --- 4. CUMULATIVE SUM & THRESHOLD ---
        # Calculate normalized cumulative sum
        cumsum = torch.cumsum(sorted_weights, dim=0)
        total_relevance = cumsum[-1]
        normalized_cumsum = cumsum / total_relevance

        # Find the first index where we cross the threshold
        # (searchsorted finds the insertion point, which corresponds to our cut)
        cutoff_idx = torch.searchsorted(normalized_cumsum, threshold).item()

        # We need to include the index that crossed the threshold, so we slice up to cutoff_idx + 1
        indices_to_keep = sorted_indices[:cutoff_idx + 1]

        # Convert to standard python list for Quimb
        indices_list = indices_to_keep.tolist()


        _, nodes = self.count_trainable_nodes_on_bond(bond_tag)

        for node in nodes:
            # Slice 'mu' (Single index)
            self.fancy_isel(self.mu[node], {f'{bond_tag}': indices_list})
            
            # Slice 'sigma' (Two indices: bond_tag AND bond_tag_prime)
            self.fancy_isel(self.sigma[node], {
                f'{bond_tag}': indices_list, 
                f'{bond_tag}_prime': indices_list
            })

        if bond_tag in self.p_bonds:
            self.p_bonds[bond_tag].trim(indices_list)

        if bond_tag in self.q_bonds:
            self.q_bonds[bond_tag].trim(indices_list)

        if verbose:
            print(f"Original Weights: {weights}")
            print(f"Sorted Indices:   {sorted_indices}")
            print(f"Keeping Indices:  {indices_list}")
            print(f"Number of remaining indices {len(indices_list)}, Number of cut Indices {len(weights) - len(indices_list)}")
        return
  
    def fancy_isel(self, tensor, index_dict):
        # We work on a copy of the data first to avoid mutating mid-process
        # (Though we will modify the tensor in-place at the end)
        new_data = tensor.data
    
        # We iterate over the requested slices
        # e.g., {'b1': [0, 2], 'b1_prime': [0, 2]}
        for index_name, keep_indices in index_dict.items():
        
            # 1. Find which axis this name corresponds to
            try:
                axis_idx = tensor.inds.index(index_name)
            except ValueError:
                raise ValueError(f"Index '{index_name}' not found in tensor indices: {tensor.inds}")

            # 2. Build the slicer
            # We start with [slice(None), slice(None), ...] (select everything)
            slicer = [slice(None)] * new_data.ndim
        
            # Replace the target axis with the list of indices to keep
            slicer[axis_idx] = keep_indices
        
            # 3. Apply the slice
            # Note: By doing this sequentially inside the loop, we avoid the 
            # "Advanced Indexing" trap where x[[0,1], [0,1]] returns the diagonal.
            # This ensures we get the full sub-block (orthogonal slicing).
            new_data = new_data[tuple(slicer)]

        # 4. Update the tensor in-place
        # modify() tells quimb "Here is the new data, please trust me that 
        # the indices (tags) haven't changed, only the shape size."
        tensor.modify(data=new_data)
    
        return tensor
    def trim_bonds(self, verbose = False):
        for bond in self.q_bonds:
            if bond in self.output_dimensions or bond in self.input_indices:
                continue
            self._trim_bond(bond, verbose = verbose)
        return
        
    def _unprime_indices_tensor(self, tensor: qt.Tensor, prime_suffix: str = "_prime") -> qt.Tensor:

        reindex_map = {
            ind: ind[:-len(prime_suffix)]
            for ind in tensor.inds
            if ind.endswith(prime_suffix)
        }
        return tensor.reindex(reindex_map)

    def fit(self, epochs, track_elbo=False, trim_every_epochs = None):
        """
        Train the model for a given number of epochs.
        
        Args:
            epochs: Number of training iterations
            track_elbo: If True, compute and print ELBO at each epoch (slower but informative)
        """
        bonds = [i for i in self.mu.ind_map if i != self.batch_dim]
        nodes = list(self.mu.tag_map.keys())
        if not isinstance(trim_every_epochs, int) or not isinstance(trim_every_epochs, bool):
            raise ValueError(f"trim_every_epochs must be a Bool, or an int to tell the frequency, instead got {trim_every_epochs}")

        if not isinstance(trim_every_epochs, int):
            if trim_every_epochs:
                trim_every_epochs = 1
            else:
                trim_every_epochs = epochs + 1
        
        # Compute initial ELBO if tracking
        if track_elbo:
            elbo_before = self.compute_elbo(verbose=False, print_components=False)
            print(f"Initial ELBO (before training): {elbo_before:.4f}")
            print("-" * 60)
        
        for i in range(epochs):
            # Store ELBO before updates
            if track_elbo:
                elbo_before_epoch = self.compute_elbo(verbose=False, print_components=False)
            
            # Perform all updates for this epoch
            for node_tag in nodes:
                self.update_sigma_node(node_tag)
                self.update_mu_node(node_tag)
            for bond_tag in bonds:
                self.update_bond(bond_tag)
            self.update_tau()
            
            # Print progress
            ctau = self.q_tau.concentration
            rtau = self.q_tau.rate
            mse = self._calc_mu_mse()/self.data.samples
            if epochs % trim_every_epochs == 0:
                self.trim_bonds()
            if track_elbo:
                elbo_after_epoch = self.compute_elbo(verbose=False, print_components=False)
                delta_elbo = elbo_after_epoch - elbo_before_epoch
                status = "✓" if delta_elbo >= 0 else "✗"
                print(f"Epoch {i+1}/{epochs} | MSE {mse.data:.4f}, E[τ] {self.get_tau_mean():.2f}, "
                      f"ELBO {elbo_after_epoch:.4f} (Δ={delta_elbo:+.4f}) {status}")
            else:
                print(f"MSE {mse.data:.4f}, E[t] {self.get_tau_mean()}, C = {ctau}, R = {rtau:.4f}")
        
        if track_elbo:
            print("-" * 60)
            print(f"Final ELBO: {elbo_after_epoch:.4f}")
            print(f"Total change: {elbo_after_epoch - elbo_before:+.4f}")
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

