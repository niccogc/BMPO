# type: ignore
import itertools
import math
import numpy as np
import quimb.tensor as qt
from typing import List, Dict, Tuple, Any, Optional

from tensor.distributions import (
    GammaDistribution, 
    MultivariateGaussianDistribution, 
    ProductDistribution
)

class Inputs:
    """
        input is a list of lenght number of batches, where there is a dictionary of what the inputs are connecting for and which labels TODO: make a method that batches and labels auto.
    """
    def __init__(self, inputs: List[any],
                 outputs: List[any],
                 outputs_labels: List[str],
                 input_labels: List[str],
                 batch_dim: str = "s",
                 batch_size = None):

        self.batch_size = inputs[0].shape[0] if batch_size is None else batch_size
        self.samples = outputs[0].shape[0]
        self.batch_dim = batch_dim
        self.repeated = (len(inputs) == 1)
        self.outputs_labels = outputs_labels
        self.input_labels = input_labels
        print("It is assumed that the Inputs and outputs in list are ordered as labels")
        raw_batches = self.batch_splits(inputs, outputs[0], self.batch_size)
        self.mu_sigma_y_batches = self.create_node_from_batch_splits(raw_batches)
        
        
    def batch_splits(self, xs, y, B):
        s = y.shape[0]
        for i in range(0, s, B):
            tensor = qt.Tensor(
                data=y[i:i+B],
                inds=(self.batch_dim, *self.outputs_labels),
                tags={'output'}
            )
            batch ={f"{j}": x[i:i+B] for j, x in zip(self.input_labels[:len(xs)],xs)} 
            yield batch, tensor

    def create_node_from_batch_splits(self, batch_generator):
            # Unpack the tuple yielded by batch_splits
            for input_dict, y_tensor in batch_generator:
            
                # Process only the input dictionary
                if self.repeated:
                    mu, sigma = self.prepare_inputs_batch_repeated(input_dict)
                else:
                    mu, sigma = self.prepare_inputs_batch(input_dict)
            
                yield mu, sigma, y_tensor

    def prepare_inputs_batch(self, input_data: Dict[str, any]) -> List[qt.Tensor]:
        """
        Prepare input tensors for forward pass through the network.
        
        Args:
            input_data: Dictionary mapping input index names to data arrays (numpy, torch, jax).
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
            # Scenario 2: Different data per node
            input_data = {'x1': data_1, 'x2': data_2}
            mu_inputs = btn.prepare_inputs(input_data, for_sigma=False)
            # Creates: x1(s, x1, data_1), x2(s, x2, data_2)
            
            # For sigma network (doubles inputs):
            sigma_inputs = btn.prepare_inputs(input_data, for_sigma=True)
            # Creates: x1(s, x1, data), x1_prime(s, x1_prime, data), x2(s, x2, data), x2_prime(s, x2_prime, data)
        """
        # Use input_indices from class (either provided or auto-detected in __init__)
        tensors_mu = []
        tensors_sigma = []
        for k,v in input_data.items():
            
            # Assert correct shape (must be 2D: batch x features)
            assert v.ndim == 2, f"Input data for '{k}' must be 2D (batch_size, features), got shape {v.shape}"
            
            # Create tensor with indices (batch_dim, input_idx)
            tensor = qt.Tensor(
                data=v,
                inds=(self.batch_dim, k),
                tags={f'input_{k}'}
            )
            tensors_mu.append(tensor)
            # If for_sigma, also create the prime version with the same data
            prime_idx = f"{k}_prime"
            tensor_prime = qt.Tensor(
                data=v,  # Points to same data
                inds=(self.batch_dim, prime_idx),
                tags={f'input_{prime_idx}'}
            )
            tensors_sigma.append(tensor_prime)
        
        return tensors_mu, tensors_mu + tensors_sigma

    def prepare_inputs_batch_repeated(self, input_data: Dict[str, any]) -> List[qt.Tensor]:
        """
        Prepare input tensors for forward pass through the network.
        
        Args:
            input_data: Dictionary mapping input index names to data arrays (numpy, torch, jax).
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
            # For sigma network (doubles inputs):
            sigma_inputs = btn.prepare_inputs(input_data, for_sigma=True)
            # Creates: x1(s, x1, data), x1_prime(s, x1_prime, data), x2(s, x2, data), x2_prime(s, x2_prime, data)
        """
        # Use input_indices from class (either provided or auto-detected in __init__)
        # Determine if we have single input for all or separate inputs
        # Scenario 1: Single input data for all input nodes

        input_indices = self.input_labels
        single_key = list(input_data.keys())[0]
        data = input_data[single_key]
        
        # Assert correct shape (must be 2D: batch x features)
        assert data.ndim == 2, f"Input data must be 2D (batch_size, features), got shape {data.shape}"
        
        batch_size, feature_dim = data.shape
        
        # Create tensors for each input index pointing to the SAME data
        tensors_mu = []
        tensors_sigma = []
        for input_idx in input_indices:
            # Create tensor with indices (batch_dim, input_idx)
            # All tensors point to the same data object - no copy needed
            tensor = qt.Tensor(
                data=data,
                inds=(self.batch_dim, input_idx),
                tags={f'input_{input_idx}'}
            )
            tensors_mu.append(tensor)
            # If for_sigma, also create the prime version
            prime_idx = f"{input_idx}_prime"
            tensor_prime = qt.Tensor(
                data=data,  # Points to same data
                inds=(self.batch_dim, prime_idx),
                tags={f'input_{prime_idx}'}
            )
            tensors_sigma.append(tensor_prime)
       
        return tensors_mu, tensors_mu + tensors_sigma

    @property
    def data_mu(self):
        """Yields only (mu, sigma), discarding y."""
        for mu, _, _ in self.mu_sigma_y_batches:
            yield mu

    @property
    def data_sigma(self):
        """Yields only (mu, sigma), discarding y."""
        for _, sigma, _ in self.mu_sigma_y_batches:
            yield sigma

    @property
    def data_y(self):
        """Yields only (mu, sigma), discarding y."""
        for _, _, y in self.mu_sigma_y_batches:
            yield y

    @property
    def data_mu_y(self):
        """Yields only (mu, sigma), discarding y."""
        for mu, _, y in self.mu_sigma_y_batches:
            yield mu, y
    # Usage:
    # for mu, sigma in loader.inputs_only:
    #     ...

    def __str__(self):
            """
            Allows calling print(loader_instance). 
            Peeks at the first batch to show structure without consuming the data.
            """
            # 1. Tee the generator: 'view' is for printing, 'save' keeps the data safe
            gen_view, self.mu_sigma_y_batches = itertools.tee(self.mu_sigma_y_batches)
        
            try:
                # 2. Peek at the first batch
                first_batch = next(gen_view)
                mu, sigma, y = first_batch
            
                # 3. Format the info
                mu_inds = [list(t.inds) for t in mu]
                sigma_inds = [list(t.inds) for t in sigma]
            
                # 4. Build the string
                header = (
                    f"\n>>> InputLoader Summary (Batch Size: {self.batch_size}, Samples: {self.samples}, Batches Number: {math.ceil(self.samples/ self.batch_size)})\n"
                    f"{'TYPE':<8} | {'SHAPE':<15} | {'INDICES'}\n"
                    f"{'-'*60}\n"
                )
            
                row_y = f"{'Target':<8} | {str(y.shape):<15} | {y.inds}\n"
                row_mu = f"{'Mu':<8} | {str(mu[0].shape):<15} | {mu_inds} ... ({len(mu)} tensors)\n"
                row_sig = f"{'Sigma':<8} | {str(sigma[0].shape):<15} | {sigma_inds} ... ({len(sigma)} tensors)"
            
                return header + row_y + row_mu + row_sig
            
            except StopIteration:
                return "InputLoader (Empty/Exhausted)"        

class BTNBuilder:
    """
    Builder class for constructing the Bayesian Tensor Network components.
    """

    def __init__(self, mu: qt.TensorNetwork, output_dimensions: List[str], batch_dim: str = "s"):
        self.mu = mu
        self.output_dimensions = output_dimensions
        self.batch_dim = batch_dim
        
        self.p_bonds: Dict[str, GammaDistribution] = {}
        self.p_nodes: Dict[str, MultivariateGaussianDistribution] = {} 
        
        self.q_bonds: Dict[str, GammaDistribution] = {}
        self.q_nodes: Dict[str, MultivariateGaussianDistribution] = {}
        
        self.sigma_tn: Optional[qt.TensorNetwork] = None

    def build_model(self) -> Tuple[Dict, Dict, Dict, Dict, qt.TensorNetwork]:
        all_inds = self.mu.ind_map
        self._build_edge_distributions(all_inds)
        self.sigma_tn = self._construct_sigma_topology()
        self._build_node_distributions()
        return self.p_bonds, self.p_nodes, self.q_bonds, self.q_nodes, self.sigma_tn

    def _build_edge_distributions(self, all_inds_map: Dict[str, Any]):
        """
        Constructs Gamma distributions with quimb.Tensor parameters.
        Tags: {ind}_alpha and {ind}_beta.
        """

        # TODO: Wait It seems somenthing is wrong, we define p_alpha and q_alpha with concentration etc for VECTOR which is right, of lenght dimension. But doesnt GammaDistribution handles Single Gamma Distributions??? the Full distribution of the bond is the product of the Gamma distributions in its dimension! It doesnt seem that is what is happening. We need to check how distributions work.
        for ind, tids in all_inds_map.items():
            if ind in self.output_dimensions or ind == self.batch_dim:
                continue
            
            exemplar_tensor = self.mu.tensor_map[next(iter(tids))]
            dim_size = exemplar_tensor.shape[exemplar_tensor.inds.index(ind)]
            
            # --- Build Prior (p) ---
            # Tag convention: bondname_alpha, bondname_beta
            p_alpha = qt.Tensor(data=np.ones(dim_size), inds=(ind,), tags={f"{ind}_alpha", "prior"})
            p_beta = qt.Tensor(data=np.ones(dim_size), inds=(ind,), tags={f"{ind}_beta", "prior"})
            
            self.p_bonds[ind] = GammaDistribution(concentration=p_alpha, rate=p_beta)
            
            # --- Build Posterior (q) ---
            q_alpha = qt.Tensor(data=np.ones(dim_size), inds=(ind,), tags={f"{ind}_alpha", "posterior"})
            q_beta = qt.Tensor(data=np.ones(dim_size), inds=(ind,), tags={f"{ind}_beta", "posterior"})
            
            self.q_bonds[ind] = GammaDistribution(concentration=q_alpha, rate=q_beta)

    def _construct_sigma_topology(self) -> qt.TensorNetwork:
        sigma_tensors = []
        for tensor in self.mu:
            original_inds = tensor.inds
            original_tags = tensor.tags
            
            non_output_inds = []
            output_inds = []
            
            for ix in original_inds:
                if ix in self.output_dimensions:
                    output_inds.append(ix)
                else:
                    non_output_inds.append(ix)
            
            non_output_primes = [f"{ix}_prime" for ix in non_output_inds]
            sigma_inds = tuple(non_output_inds + non_output_primes + output_inds)
            
            shape_map = dict(zip(tensor.inds, tensor.shape))
            new_shape = []
            for ix in sigma_inds:
                if ix.endswith('_prime') and ix not in shape_map:
                    original_name = ix[:-6] 
                else:
                    original_name = ix
                new_shape.append(shape_map[original_name])
            
            sigma_data = np.random.randn(*new_shape) * 0.01
            sigma_tags = {f"{tag}_sigma" for tag in original_tags}

            sigma_tensor = qt.Tensor(
                data=sigma_data,
                inds=sigma_inds,
                tags=sigma_tags
            )
            sigma_tensors.append(sigma_tensor)
            
        return qt.TensorNetwork(sigma_tensors)

    def _build_node_distributions(self):
        for tensor in self.mu:
            node_key = tuple(sorted(list(tensor.tags)))
            
            sigma_tensor = None
            for tag in tensor.tags:
                sigma_tag = f"{tag}_sigma"
                try:
                    candidates = self.sigma_tn[sigma_tag]
                    if isinstance(candidates, qt.Tensor):
                        sigma_tensor = candidates
                        break
                except KeyError:
                    continue
            
            if sigma_tensor is None:
                raise ValueError(f"Could not find corresponding Sigma tensor for node {tensor.tags}")

            self.q_nodes[node_key] = MultivariateGaussianDistribution(
                loc=tensor,
                covariance_matrix=sigma_tensor
            )
            
            prior_loc = tensor.copy()
            prior_loc.modify(data=np.zeros_like(tensor.data))
            
            self.p_nodes[node_key] = MultivariateGaussianDistribution(
                loc=prior_loc,
                covariance_matrix=None 
            )
