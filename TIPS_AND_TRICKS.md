# Small smart ideas

  **Computation**
  - Note that, the contraction of the network to get the forward, to calculate MSE and everything is given by. Node_i * T_i Tensor_network. basically the node by the environment produces the forward. So when during the training loop. If the environment is needed (training mode), first compute that, then do the block updates, then compute the forward by simply contracting the environment with the node.

  - Note, the update of beta Tau can reuse a lot of computation, it is simply the total MSE over the samples + the sum over all samples of the covariance forward, that during training can be easily computed with environment_sum_forward. (Check THEORETICAL_MODEL.md)

  **Therefore we need three types of forward**
  1. simple_forward: just do the contraction of the nodes with batches and return the sample dimensions x output_dimension
  2. environment_forward: needs an environment for all the samples, and a block, and just perform the contraction between the environment for all samples and the block.
  3. environment_sum_forward: needs the sum over all samples of the environment, the block, and compute the contraction with the block (used mostly during the training process for the sigma matrix where we dont really need sample by sample output)
