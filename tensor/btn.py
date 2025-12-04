# type: ignore
from typing import List
import quimb.tensor as qt
import numpy as np

class BTN:
    def __init__(self,
                 mu: qt.TensorNetwork,
                 sigma: qt.TensorNetwork,
                 fixed_nodes: List[str],
                 output_dimensions: List[str],
                 batch_dim: str = "s"
                 ):
        self.mu = mu
        self.sigma = sigma
        self.fixed_nodes = fixed_nodes
        self.output_dimensions = output_dimensions
        self.batch_dim = batch_dim

    def _batch_forward(self, tn: qt.TensorNetwork, inputs: List[qt.Tensor], output_inds: List[str]) -> qt.Tensor:
        full_tn = tn & inputs
        return full_tn.contract(output_inds=output_inds)

    def forward(self, tn: qt.TensorNetwork, inputs: List[List[qt.Tensor]]) -> qt.Tensor:
        # Define the strict order we want: Batch dimension FIRST
        target_inds = [self.batch_dim] + self.output_dimensions
        
        batch_results = []
        for batch_tensors in inputs:
            res = self._batch_forward(tn, batch_tensors, output_inds=target_inds)
            
            # 🚨 CRITICAL FIX: Explicitly transpose the tensor to match 'target_inds'
            # This ensures the data shape is (s, y1, y2...) so axis=0 is correct.
            res.transpose_(*target_inds) 
            
            batch_results.append(res)

        # Now axis=0 corresponds to 's', so concatenation works
        final_data = np.concatenate([t.data for t in batch_results], axis=0)
        
        # Use indices from the first result (which we just transposed correctly)
        final_tensor = qt.Tensor(final_data, inds=batch_results[0].inds)

        return final_tensor

# type: ignore
from typing import List
import quimb.tensor as qt
import numpy as np

class BTN:
    def __init__(self,
                 output_dimensions: List[str],
                 batch_dim: str = "s"
                 ):
        self.output_dimensions = output_dimensions
        self.batch_dim = batch_dim

    def _batch_forward(self, tn: qt.TensorNetwork, inputs: List[qt.Tensor], output_inds: List[str]) -> qt.Tensor:
        full_tn = tn & inputs
        return full_tn.contract(output_inds=output_inds)

    def forward(self, tn: qt.TensorNetwork, inputs: List[List[qt.Tensor]]) -> qt.Tensor:
        target_inds = [self.batch_dim] + self.output_dimensions
        
        batch_results = []
        for batch_tensors in inputs:
            res = self._batch_forward(tn, batch_tensors, output_inds=target_inds)
            res.transpose_(*target_inds) 
            batch_results.append(res)

        final_data = np.concatenate([t.data for t in batch_results], axis=0)
        final_tensor = qt.Tensor(final_data, inds=batch_results[0].inds)

        return final_tensor

    def get_environment(self, tn: qt.TensorNetwork, target_tag: str, copy = True) -> qt.Tensor:
        """
        Calculates the environment.
        Indices = (Bonds connecting to Target) + (Global Outputs of other nodes)
        """
        # 1. Identify Target and its original indices
        target_node = tn[target_tag]
        if isinstance(target_node, qt.TensorNetwork):
             original_inds = target_node.outer_inds()
        else:
             original_inds = target_node.inds

        # 2. Create the "hole"
        if copy:
            env_tn = tn.copy()
        else:
            env_tn = tn
        env_tn.delete(target_tag)

        # 3. Determine 'Hole' indices (Bonds shared with the deleted node)
        # We look for indices that were on the target AND exist in the remaining network map.
        shared_bonds = [ix for ix in original_inds if ix in env_tn.ind_map]
        # B. Global indices (Outputs & Batch Dim) that must not be summed over
        # We check if they exist in the remaining network before asking to keep them.
        global_inds_to_check = self.output_dimensions + [self.batch_dim]
        global_inds = [ix for ix in global_inds_to_check if ix in env_tn.ind_map]
        
        final_env_inds = list(set(shared_bonds + global_inds))

        # 4. Contract
        env_tensor = env_tn.contract(output_inds=final_env_inds)
        
        return env_tensor

