"""
Manually compute Node2 KL to debug why it explodes.
"""
# type: ignore
import torch
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
import numpy as np

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)

# Setup
N = 500
x_raw = 2 * torch.rand(N, 1) - 1
y_raw = torch.cat([2*(x_raw**3), x_raw**2, -x_raw**3], dim=1) + 0.05 * torch.randn(N, 3)
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)

loader = Inputs(
    inputs=[x_features], 
    outputs=[y_raw], 
    outputs_labels=['y'], 
    input_labels=['x1', 'x2', 'x3'], 
    batch_dim='s', 
    batch_size=100
)

def init_weights(shape):
    w = torch.randn(*shape, dtype=torch.float64)
    return w/torch.norm(w)

t1 = qt.Tensor(data=init_weights((2, 4, 4)), inds=('x1', 'b1', 'b3'), tags={'Node1'})
t2 = qt.Tensor(data=init_weights((4, 2, 4)), inds=('b1', 'x2', 'b2'), tags={'Node2'})
t3 = qt.Tensor(data=init_weights((4, 2, 4, 3)), inds=('b2', 'x3', 'b3', 'y'), tags={'Node3'})

mu_tn = qt.TensorNetwork([t1, t2, t3])
model = BTN(mu=mu_tn, data_stream=loader, batch_dim='s', method='cholesky')

# Do TWO update cycles to get to epoch 2 (where it explodes)
for epoch in range(2):
    for node_tag in ['Node1', 'Node2', 'Node3']:
        model.update_sigma_node(node_tag)
        model.update_mu_node(node_tag)
    bonds = [i for i in model.mu.ind_map if i != model.batch_dim]
    for bond_tag in bonds:
        model.update_bond(bond_tag)
    model.update_tau()

print('='*80)
print('DEBUGGING NODE2 KL COMPUTATION (after epoch 1)')
print('='*80)

# First, find the correct node key
print('\nAvailable node keys in p_nodes:')
for key in model.p_nodes.keys():
    print(f'  {key}')

node_tag = 'Node2'
# Get the correct key: tuple of sorted tags
node_key = tuple(sorted(list(model.mu[node_tag].tags)))
print(f'\nNode2 tag: {node_tag}')
print(f'Node2 key for p_nodes: {node_key}')

print(f'\nNode2 structure:')
print(f'  mu.inds: {model.mu[node_tag].inds}')
print(f'  mu.shape: {model.mu[node_tag].shape}')
print(f'  sigma.inds: {model.sigma[node_tag].inds}')
print(f'  sigma.shape: {model.sigma[node_tag].shape}')

# Get the indices
variational_mu, output_mu, mu_idx = model.get_variational_outputs_inds(model.mu[node_tag])
print(f'\nIndices:')
print(f'  variational_mu: {variational_mu}')
print(f'  output_mu: {output_mu}')
print(f'  mu_idx: {mu_idx}')

# Compute theta (prior)
print(f'\n--- Computing PRIOR (theta) ---')
theta = model.theta_block_computation(node_tag, use_prior=True)
print(f'theta before new_ind_pair_diag:')
print(f'  inds: {theta.inds}')
print(f'  shape: {theta.shape}')
print(f'  data range: [{theta.data.min():.4f}, {theta.data.max():.4f}]')

# Apply new_ind_pair_diag
theta_expanded = theta.copy()
for inds in mu_idx:
    primed = inds + '_prime'
    theta_expanded.new_ind_pair_diag(inds, inds, primed, inplace=True)

print(f'\ntheta after new_ind_pair_diag:')
print(f'  inds: {theta_expanded.inds}')
print(f'  shape: {theta_expanded.shape}')

primed_mu_idx = [inds + '_prime' for inds in mu_idx]
print(f'  mu_idx: {mu_idx}')
print(f'  primed_mu_idx: {primed_mu_idx}')

prior_precision = theta_expanded.to_dense(mu_idx, primed_mu_idx)
print(f'\nprior_precision:')
print(f'  shape: {prior_precision.shape}')
print(f'  Expected: ({int(np.prod(model.mu[node_tag].shape))}, {int(np.prod(model.mu[node_tag].shape))})')

# FIXED: Invert precision to get covariance
prior_cov = torch.linalg.inv(prior_precision)
print(f'\nprior_cov (after inversion):')
print(f'  shape: {prior_cov.shape}')

# Check prior_cov properties
eigs = torch.linalg.eigvalsh(prior_cov)
print(f'  Eigenvalues: min={eigs.min():.4e}, max={eigs.max():.4e}')
print(f'  Condition number: {(eigs.max()/eigs.min()).item():.2e}')
print(f'  Is positive definite: {(eigs.min() > 0).item()}')

prior_loc = torch.zeros_like(model.mu[node_tag].data.flatten())
print(f'\nprior_loc:')
print(f'  shape: {prior_loc.shape}')

# Compute q (posterior)
print(f'\n--- Computing POSTERIOR (q) ---')
mu_flat = model.mu[node_tag].data.flatten()
print(f'mu_flat:')
print(f'  shape: {mu_flat.shape}')
print(f'  range: [{mu_flat.min():.4f}, {mu_flat.max():.4f}]')

# Expand sigma for output dimensions if needed
q_cov = model.sigma[node_tag].copy()
print(f'\nsigma before expanding outputs:')
print(f'  inds: {q_cov.inds}')
print(f'  shape: {q_cov.shape}')

for out in output_mu:
    primed = out + '_prime'
    print(f'  Adding diagonal for output: {out}, {primed}')
    q_cov = q_cov.new_ind_pair_diag(out, out, primed)

print(f'\nsigma after expanding outputs:')
print(f'  inds: {q_cov.inds}')
print(f'  shape: {q_cov.shape}')

q_cov_dense = q_cov.to_dense(mu_idx, primed_mu_idx)
print(f'\nq_cov (posterior covariance):')
print(f'  shape: {q_cov_dense.shape}')

# Check q_cov properties
eigs_q = torch.linalg.eigvalsh(q_cov_dense)
print(f'  Eigenvalues: min={eigs_q.min():.4e}, max={eigs_q.max():.4e}')
print(f'  Condition number: {(eigs_q.max()/eigs_q.min()).item():.2e}')
print(f'  Is positive definite: {(eigs_q.min() > 0).item()}')

# Compute KL
print(f'\n--- Computing KL Divergence ---')
try:
    p = model.p_nodes[node_key].forward(loc=prior_loc, cov=prior_cov)
    q = model.q_nodes[node_key].forward(loc=mu_flat, cov=q_cov_dense)
    kl = torch.distributions.kl_divergence(q, p).item()
    print(f'KL divergence: {kl:.4f}')
    
    if kl > 1000:
        print(f'\n⚠️  KL is very large! This suggests a problem.')
        
        # Check KL formula components
        # KL = 0.5 * [tr(Σ_p^-1 Σ_q) + (μ_p - μ_q)^T Σ_p^-1 (μ_p - μ_q) - k + log(det(Σ_p)/det(Σ_q))]
        k = mu_flat.shape[0]
        
        # Mahalanobis distance
        diff = mu_flat - prior_loc
        prec_p = torch.linalg.inv(prior_cov)
        mahal = (diff @ prec_p @ diff).item()
        
        # Trace term
        trace_term = torch.trace(prec_p @ q_cov_dense).item()
        
        # Log det term
        logdet_p = torch.logdet(prior_cov).item()
        logdet_q = torch.logdet(q_cov_dense).item()
        
        print(f'\nKL components:')
        print(f'  k (dimension): {k}')
        print(f'  Mahalanobis distance: {mahal:.4f}')
        print(f'  Trace term: {trace_term:.4f}')
        print(f'  Log det prior: {logdet_p:.4f}')
        print(f'  Log det posterior: {logdet_q:.4f}')
        print(f'  Log det difference: {logdet_p - logdet_q:.4f}')
        
        kl_manual = 0.5 * (trace_term + mahal - k + logdet_p - logdet_q)
        print(f'  KL (manual): {kl_manual:.4f}')
        
except Exception as e:
    print(f'ERROR computing KL: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '='*80)
