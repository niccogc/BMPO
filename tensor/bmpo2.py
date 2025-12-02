import torch
import quimb.tensor as qtn
import numpy as np
import copy
from tqdm.auto import tqdm

class BayesianQuimbTN:
    def __init__(self, mu_tn: qtn.TensorNetwork, rank_indices=None):
        """
        Args:
            mu_tn: The initialized quimb TensorNetwork representing the Mean (Mu).
            rank_indices: Set of indices to be treated as probabilistic bonds. 
                          If None, all internal indices are selected.
        """
        # 1. Mu Network
        self.mu_tn = mu_tn.copy()
        
        # 2. Identify Bonds (Rank Indices)
        if rank_indices is None:
            self.rank_indices = set(self.mu_tn.inner_inds())
        else:
            self.rank_indices = set(rank_indices)
            
        # 3. Sigma Network (Squared structure)
        # We create a copy and rename indices to avoid collision
        self.sigma_tn = mu_tn.copy()
        self._init_sigma_structure()
        
        # 4. Global Noise (Tau)
        self.tau_alpha = torch.tensor(2.0)
        self.tau_beta = torch.tensor(1.0)
        
        # 5. Bond Priors (Gamma)
        # Map: index_name -> {'alpha': tensor, 'beta': tensor}
        self.bond_params = {}
        self._init_bond_priors()

        # Cache for bond expectations to avoid recomputing constantly
        self.bond_expectations = {}

    @property
    def expected_tau(self):
        return self.tau_alpha / self.tau_beta

    def _init_sigma_structure(self):
        """
        Initialize Sigma Nodes following.
        Logic: Cov = sigma^2 * I
        Scaling: sigma^2 = 1.0 / (num_nodes * avg_rank)
        """
        # Calculate stats for scaling
        num_nodes = self.sigma_tn.num_tensors
        ranks = [self.sigma_tn.ind_size(i) for i in self.rank_indices]
        avg_rank = np.mean(ranks) if ranks else 1.0
        initial_variance = 1.0 / (num_nodes * avg_rank)
        
        # Rename indices to denote squared space (d -> d^2)
        # We map ind -> ind_sq to keep them distinct from mu network
        self.sq_ind_map = {i: f"{i}_sq" for i in self.sigma_tn.ind_map}
        self.sigma_tn.reindex_(self.sq_ind_map)
        
        # Initialize each tensor
        for t in self.sigma_tn:
            # Flatten to (d_total,) to create diagonal
            d_total = t.data.numel()
            
            # Create diagonal covariance: sigma^2 * I
            # Note: We store the FLATTENED version of the squared tensor if possible,
            # or reshape it to match the (d1^2, d2^2...) structure.
            # Here we reshape to (d1^2, d2^2...) to maintain graph structure.
            
            # 1. Create Diagonal (d_total, d_total)
            # Optimization: We only store the diagonal elements if we treat it as a vector,
            # but standard TN contraction expects full connectivity. 
            # We initialize with the diagonal injected into the reshaped tensor.
            
            # Construct the tensor that corresponds to Identity on the flattened scale
            # shape: (d1, d2..., d1, d2...) -> flattened -> diag -> reshaped
            # Ideally, quimb handles dense tensors.
            
            # Approximation for initialization: 
            # We want E[W x W] = mu x mu + variance. 
            # The sigma_tn stores the Variance part.
            
            # Create the tensor with correct shape (d1^2, d2^2...)
            new_shape = tuple(d**2 for d in t.shape)
            
            # Create identity in (d_total, d_total) then reshape
            ident = torch.eye(d_total) * initial_variance
            
            # We need to map (d, d) -> (d1...dn, d1...dn) -> (d1 d1, d2 d2...)
            # This permutation is tricky.
            # Let's use the same logic as _initialize_sigma_nodes in bayesian_mpo.py
            # 1. Reshape to (d1, d2..., d1, d2...)
            expanded_shape = t.shape + t.shape
            sigma_expanded = ident.reshape(expanded_shape)
            
            # 2. Permute to (d1, d1, d2, d2...)
            # dimensions: 0, 1, ... N-1  and  N, N+1, ... 2N-1
            # We want: 0, N, 1, N+1, ...
            n_dims = len(t.shape)
            perm = []
            for i in range(n_dims):
                perm.append(i)
                perm.append(i + n_dims)
            
            sigma_permuted = sigma_expanded.permute(*perm)
            
            # 3. Flatten pairs (d1, d1) -> d1^2
            sigma_reshaped = sigma_permuted.reshape(new_shape)
            
            t.modify(data=sigma_reshaped)

    def _init_bond_priors(self):
        """Initialize Gamma(2.0, 1.0) for all ranks."""
        for ind in self.rank_indices:
            size = self.mu_tn.ind_size(ind)
            self.bond_params[ind] = {
                'alpha': torch.ones(size) * 2.0,
                'beta': torch.ones(size) * 1.0
            }
            self.bond_expectations[ind] = self.bond_params[ind]['alpha'] / self.bond_params[ind]['beta']

    def _get_theta_tensor(self, tid):
        """
        Construct Theta tensor (Diagonal of prior expectations) for a specific node.
        Logic matches 'compute_theta_tensor' in BMPO.
        """
        t = self.mu_tn.tensor_map[tid]
        factors = []
        
        for ind in t.inds:
            if ind in self.rank_indices:
                # E[lambda]
                factors.append(self.bond_expectations[ind])
            else:
                # Physical/Input dimension: 1.0
                factors.append(torch.ones(self.mu_tn.ind_size(ind)))
        
        # Outer product: v1 x v2 x v3...
        # Einsum: a,b,c -> abc
        eq = ",".join(chr(97 + i) for i in range(len(factors))) + "->" + "".join(chr(97 + i) for i in range(len(factors)))
        theta = torch.einsum(eq, *factors)
        return torch.diag(theta.flatten())

    def update_block_variational(self, tid, X_dict, y):
        """
        Step 1: Update Node Distributions.
        """
        # --- 1. Compute J_mu (Environment of Mu) ---
        # We clamp the inputs and contract everything EXCEPT the target node
        # This gives us T_i * mu * x
        
        # Create a clamped network
        net_clamped = self.mu_tn.copy()
        
        # Attach data to physical indices
        # We assume X_dict maps index_name -> (Batch, Dim) tensor
        batch_size = y.shape[0]
        for ind, data in X_dict.items():
            # Create a data tensor connected to the physical index
            # data shape: (Batch, Dim)
            # Ind needs to be contracted. 
            data_tensor = qtn.Tensor(data=data, inds=['batch', ind])
            net_clamped &= data_tensor
            
        # Remove the target tensor
        target_tensor = net_clamped.tensor_map[tid]
        net_clamped.delete_tensor(tid)
        
        # Contract. Output indices should be the target's indices + 'batch'
        # This is J_mu(x_n)
        out_inds = ('batch',) + target_tensor.inds
        J_mu_tensor = net_clamped.contract(output_inds=out_inds)
        
        # Flatten J_mu to (Batch, D_total)
        # Note: quimb output order matches out_inds. 
        # We keep batch as dim 0
        J_mu_flat = J_mu_tensor.data.reshape(batch_size, -1)
        
        # --- 2. Compute J_sigma (Environment of Sigma) ---
        # Similar process for Sigma network.
        # Inputs to sigma are technically X \otimes X.
        # But for efficiency we can just contract the sigma environment.
        # If sigma structure handles the variance flow, we just need to contract it.
        # Note: We assume the Sigma network is "closed" except for the target hole.
        # Does Sigma depend on X? Yes, typically Variance(output) depends on X.
        # For this translation, we assume we need to contract sigma with inputs too.
        
        sigma_clamped = self.sigma_tn.copy()
        
        # Attach squared data or outer product data? 
        # Based on theory: sum_n (T_i Sigma) * (x_n x_n)
        # We attach (x_n ** 2) effectively if we work in diagonal approximations,
        # or we might need the full outer product if we are strict.
        # Given we initialized sigma with squared dimensions, we attach X squared if factorized.
        # For simplicity/robustness in translation:
        for ind, data in X_dict.items():
            ind_sq = self.sq_ind_map[ind] # Get the mapped name
            # Shape (Batch, Dim). We need (Batch, Dim^2) or (Batch, Dim, Dim)
            # Let's compute the outer product per sample flattened
            # data: (B, D) -> (B, D, 1) * (B, 1, D) -> (B, D, D) -> (B, D^2)
            data_sq = torch.bmm(data.unsqueeze(2), data.unsqueeze(1)).reshape(batch_size, -1)
            
            data_t_sq = qtn.Tensor(data=data_sq, inds=['batch', ind_sq])
            sigma_clamped &= data_t_sq

        # Find the corresponding sigma tensor ID (it was copied so tags/inds match but mapped)
        # We can find it by tag or geometry. 
        # Since we copied, the tids might be different, but we can match via tags if present.
        # Fallback: maintain a mapping or assume same iteration order if not modified structure.
        # Let's assume we can Select by tags.
        tags = target_tensor.tags
        tid_sigma = tuple(self.sigma_tn.select(tags, which='any').tids)[0]
        
        sigma_clamped.delete_tensor(tid_sigma)
        
        # Contract
        target_sigma = self.sigma_tn.tensor_map[tid_sigma]
        out_inds_sq = ('batch',) + target_sigma.inds
        J_sigma_tensor = sigma_clamped.contract(output_inds=out_inds_sq)
        
        # Sum over batch (as per Eq: sum_n J_sigma)
        J_sigma_sum = J_sigma_tensor.data.sum(dim=0).flatten() # (D_total^2) or (D_total, D_total) flattened
        
        # Reshape to Matrix (D_total, D_total)
        d_total = J_mu_flat.shape[1]
        J_sigma_sum_mat = J_sigma_sum.reshape(d_total, d_total)

        # --- 3. Compute Update Terms ---
        E_tau = self.expected_tau
        Theta_diag = self._get_theta_tensor(tid)
        
        # A term: E[tau] * ( sum(J_sigma) + sum(J_mu outer J_mu) ) + Theta
        # J_mu_outer_sum = J_mu.T @ J_mu
        J_mu_outer_sum = torch.matmul(J_mu_flat.T, J_mu_flat)
        
        Precision = E_tau * (J_sigma_sum_mat + J_mu_outer_sum) + Theta_diag
        
        # b term: E[tau] * J_mu.T @ y
        # y shape (Batch,). J_mu_flat (Batch, D)
        rhs = E_tau * torch.matmul(J_mu_flat.T, y)
        
        # --- 4. Solve and Update ---
        # Covariance = Inv(Precision)
        # Use Cholesky for stability
        try:
            L = torch.linalg.cholesky(Precision)
            New_Sigma = torch.cholesky_inverse(L)
            New_Mu_flat = torch.matmul(New_Sigma, rhs)
        except:
            # Fallback
            New_Sigma = torch.linalg.inv(Precision)
            New_Mu_flat = torch.matmul(New_Sigma, rhs)
            
        # Update Mu Tensor
        target_tensor.modify(data=New_Mu_flat.reshape(target_tensor.shape))
        
        # Update Sigma Tensor
        # We need to reshape the dense (D, D) matrix back to (d1^2, d2^2...)
        # We use the inverse of the initialization permutation
        expanded_shape = target_tensor.shape + target_tensor.shape
        sigma_expanded = New_Sigma.reshape(expanded_shape)
        
        # Permute back to interleaved (d1, d1, d2, d2...)
        n_dims = len(target_tensor.shape)
        # Init perm was: 0, N, 1, N+1...
        # We need to map back to: 0, 1... N-1, N...
        # Actually, for the Sigma Tensor in the network, we want the interleaved/squared shape (d1^2, d2^2...)
        # So we need to permute from (d1..dn, d1..dn) TO (d1, d1, d2, d2) then reshape
        perm = []
        for i in range(n_dims):
            perm.append(i)
            perm.append(i + n_dims)
        
        sigma_permuted = sigma_expanded.permute(*perm)
        new_sigma_shape = tuple(d**2 for d in target_tensor.shape)
        target_sigma.modify(data=sigma_permuted.reshape(new_shape))

    def update_bond_variational(self):
        """
        Step 2: Update Bond Distributions.
        """
        for ind in self.rank_indices:
            # Find nodes connected to this index
            # In quimb, we get the tensor IDs connected to this index
            tids = list(self.mu_tn.get_tids_from_inds(ind))
            
            # Prior params
            alpha_new = self.bond_params[ind]['alpha'].clone()
            # Formula: alpha_0 + |A(i)| * dim(i) / 2 ?? 
            # Check theory: alpha = alpha_0 + dim(i)/2 * count? 
            # In bayesian_mpo.py: alpha += |A(i)| * dim(i) / 2
            # Here dim(i) is 1? Or is it the size of the bond vector?
            # Gamma is usually per-element. The code says "vectorized parameters".
            # The theory says "sum over a in A(i)".
            
            # Rate update: beta = beta_0 + 0.5 * sum_a [ tr( (Sigma+mu^2) * Theta_env ) ]
            beta_new = self.bond_params[ind]['beta'].clone()
            
            for tid in tids:
                # Local Second Moment: E[A^2] = Sigma + Mu^2
                t_mu = self.mu_tn.tensor_map[tid]
                
                # Get Sigma for this node (need to extract diagonal logic or full tensor)
                # For partial trace, we need the diagonal of the covariance relative to the bond `ind`.
                
                # Retrieve Mu and Sigma tensors
                mu_data = t_mu.data
                
                # Identify sigma tensor
                tags = t_mu.tags
                tid_sigma = tuple(self.sigma_tn.select(tags, which='any').tids)[0]
                t_sigma = self.sigma_tn.tensor_map[tid_sigma]
                
                # We need to reshape Sigma back to (d1, d1, d2, d2...) to separate dimensions
                # Or just (d1^2, d2^2...) is fine if we view it as flattened.
                
                # We need the "diagonal" of the Sigma tensor with respect to the node indices
                # To get <A^2>, we extract the diagonal elements (i, i) for every dimension.
                # Since we stored Sigma as (d1^2, d2^2...), and we initialized it as diagonal...
                # We can compute the "Diagonal Tensor" by taking the diagonal of the reshaped 
                # (d, d) blocks.
                # Helper:
                sigma_diag_vals = []
                # Iterate dims
                for i, dim_sz in enumerate(t_mu.shape):
                    # extract column i from sigma, which is dim_sz^2
                    # We want the diagonal elements
                    # Reshape this dim to (dim_sz, dim_sz) and take diag
                    # This is complex on the full tensor.
                    pass
                
                # Approximation: Since we usually use diagonal approximations for Theta,
                # let's assume we can compute (Mu^2 + Sigma_diag).
                # To get Sigma_diag from our "Squared" tensor `t_sigma`:
                # We just need to ensure we map (i*dim + i) indices.
                # For this implementation, let's assume `t_sigma` IS the variance tensor
                # reshaped to match `t_mu`. (This works if we assume diagonal covariance).
                
                # Let's extract the variance part.
                # If t_sigma is (d1^2, d2^2), we can construct a tensor (d1, d2) 
                # by taking indices [0, 1*d+1, 2*d+2...] ?
                # Actually, earlier we stored it such that it represents the variance.
                # Let's try to extract the diagonal from the squared tensor.
                
                # Let's define E_A2 = mu^2 + sigma_variance
                # We need to obtain sigma_variance from t_sigma.
                # Reshape t_sigma to (d1, d1, d2, d2...) and take diagonals.
                full_shape = []
                for s in t_mu.shape:
                    full_shape.extend([s, s])
                
                sigma_unpacked = t_sigma.data.reshape(full_shape)
                
                # Contract indices ii, jj, kk... to get the diagonal tensor
                # Einsum: aabcc... -> abc...
                ein_in = "".join([chr(97+i)*2 for i in range(t_mu.ndim)])
                ein_out = "".join([chr(97+i) for i in range(t_mu.ndim)])
                variance_tensor = torch.einsum(f"{ein_in}->{ein_out}", sigma_unpacked)
                
                E_A2 = mu_data**2 + variance_tensor
                
                # Now contract E_A2 with Theta_env
                # Theta_env is the product of E[lambda] for ALL bonds EXCEPT `ind`
                
                # Calculate Theta for this node (ALL bonds)
                # But we want to EXCLUDE `ind`.
                
                theta_factors = []
                # Ordering matches t_mu.inds
                for i, idx_name in enumerate(t_mu.inds):
                    if idx_name == ind:
                        # Exclude this bond (set to 1.0)
                        theta_factors.append(torch.ones(t_mu.shape[i]))
                    elif idx_name in self.rank_indices:
                        # Include other bond expectations
                        theta_factors.append(self.bond_expectations[idx_name])
                    else:
                        # Physical index
                        theta_factors.append(torch.ones(t_mu.shape[i]))
                        
                # Compute contraction: E_A2 * Theta_excluding_ind
                # We can just multiply elementwise if we broadcast correctly.
                # Then sum over all dimensions EXCEPT `ind`.
                
                # Construct Theta_env tensor
                eq = ",".join(chr(97 + k) for k in range(len(theta_factors))) + "->" + "".join(chr(97 + k) for k in range(len(theta_factors)))
                theta_env = torch.einsum(eq, *theta_factors)
                
                term = E_A2 * theta_env
                
                # Sum over all dims except the one corresponding to `ind`
                dims_to_sum = [i for i, name in enumerate(t_mu.inds) if name != ind]
                partial_tr = term.sum(dim=dims_to_sum)
                
                beta_new += 0.5 * partial_tr

            # Update params
            self.bond_params[ind]['beta'] = beta_new
            # Update cache
            self.bond_expectations[ind] = self.bond_params[ind]['alpha'] / beta_new

    def update_tau_variational(self, X_dict, y):
        """
        Step 3: Update Tau.
        alpha += S/2
        beta += 0.5 * ( ||y - mu_x||^2 + Tr(Sigma_x) )
        """
        S = y.shape[0]
        self.tau_alpha += S / 2.0
        
        # 1. Forward Pass Mu
        # Clamp inputs and contract
        net_mu = self.mu_tn.copy()
        for ind, data in X_dict.items():
            net_mu &= qtn.Tensor(data=data, inds=['batch', ind])
            
        # Contract to scalar/output per batch
        # Assuming single output dimension 's' or similar, or just scalar.
        # If the original network has open indices representing output, they remain.
        # Assume output is scalar per batch for regression as per theory.
        mu_out = net_mu.contract(output_inds=['batch']) # Result (Batch,)
        mu_x = mu_out.data
        
        # 2. Forward Pass Sigma (Variance)
        # Sum of variances projected to output
        # Approximate: Tr(Sigma_x) approx sum of local variances propagated?
        # Exact: Contract Sigma network with X^2
        
        net_sigma = self.sigma_tn.copy()
        for ind, data in X_dict.items():
            ind_sq = self.sq_ind_map[ind]
            # Outer product/squared data
            data_sq = torch.bmm(data.unsqueeze(2), data.unsqueeze(1)).reshape(S, -1)
            net_sigma &= qtn.Tensor(data=data_sq, inds=['batch', ind_sq])
            
        sigma_out = net_sigma.contract(output_inds=['batch'])
        sigma_x = sigma_out.data
        
        # 3. Update Beta
        # E[(y-f)^2] = (y - mu)^2 + sigma
        mse = torch.sum((y - mu_x)**2)
        var_term = torch.sum(sigma_x)
        
        self.tau_beta += 0.5 * (mse + var_term)

    def fit(self, X_dict, y, max_iter=10, verbose=True):
        """
        Coordinate Ascent Fit Loop.
        """
        if verbose:
            print("Starting Bayesian Quimb Fit...")
            
        for i in range(max_iter):
            # 1. Block Updates
            # Iterate over all tensors in mu_tn
            # We use list of tids to avoid issues if graph changes (it shouldn't)
            tids = list(self.mu_tn.tensor_map.keys())
            for tid in tids:
                # Skip data nodes if any
                if tid not in self.mu_tn.tensor_map: continue
                self.update_block_variational(tid, X_dict, y)
                
            # 2. Bond Updates
            self.update_bond_variational()
            
            # 3. Tau Update
            self.update_tau_variational(X_dict, y)
            
            if verbose:
                mse = torch.mean((self.predict(X_dict) - y)**2)
                print(f"Iter {i+1}: Tau_Exp={self.expected_tau.item():.4f}, MSE={mse.item():.4f}")

    def predict(self, X_dict):
        net = self.mu_tn.copy()
        for ind, data in X_dict.items():
            net &= qtn.Tensor(data=data, inds=['batch', ind])
        return net.contract(output_inds=['batch']).data
