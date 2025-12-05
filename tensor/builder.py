# type: ignore
import torch
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
    def __init__(self, input: List[any], outputs: List[any], outputs_labels: List[str],input_labels: List[str], batch_dim: str = "s"):
        self.batch_dim = batch_dim
        self.repeated = len(list) == 1
        self.outputs_labels = outputs_labels
        self.input_labels = input_labels if not self.repeatd else ["dummy_label"]
        print("It is assumed that the Inputs and outputs in list are ordered as labels")
        
    def batch_splits(self, xs, y, B):
        s = y.shape[0]
        for i in range(0, s, B):
            tensor = qt.Tensor(
                data=y[i:i+B],
                inds=(self.batch_dim, *self.outputs_labels),
                tags={'output'}
            )
            batch ={f"{j}": x[i:i+B] for j, x in zip(self.input_labels,xs)} 
            yield batch, tensor

    def create_node_from_batch_splits(self, batches):
        for b in batches:
            if self.repeated:
                mubatch, sigmabatch = self.prepare_inputs_batch_repeated(b)
            else:
                mubatch, sigmabatch = self.prepare_inputs_batch(b)
            yield mubatch, sigmabatch

    def create_nodes():
        pass

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
        tensors = []
        for k,v in input_data.items():
            
            # Assert correct shape (must be 2D: batch x features)
            assert v.ndim == 2, f"Input data for '{k}' must be 2D (batch_size, features), got shape {v.shape}"
            
            # Create tensor with indices (batch_dim, input_idx)
            tensor = qt.Tensor(
                data=v,
                inds=(self.batch_dim, k),
                tags={f'input_{k}'}
            )
            tensors.append(tensor)
            mu_inputs = tensors.copy()
            # If for_sigma, also create the prime version with the same data
            prime_idx = f"{k}_prime"
            tensor_prime = qt.Tensor(
                data=v,  # Points to same data
                inds=(self.batch_dim, prime_idx),
                tags={f'input_{prime_idx}'}
            )
            tensors.append(tensor_prime)
            sigma_inputs = tensors.copy()
        
        return mu_inputs, sigma_inputs

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
            
            mu_inputs = tensors.copy()
            # If for_sigma, also create the prime version
            prime_idx = f"{input_idx}_prime"
            tensor_prime = qt.Tensor(
                data=data,  # Points to same data
                inds=(self.batch_dim, prime_idx),
                tags={f'input_{prime_idx}'}
            )
            tensors.append(tensor_prime)
            sigma_inputs = tensors.copy()
       
        return mu_inputs, sigma_inputs
        
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
