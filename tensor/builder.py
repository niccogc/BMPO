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
