"""
Debug script to find why ELBO decreases (which should never happen with closed-form updates).
"""
import torch
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)

print('='*80)
print('DEBUGGING ELBO BUG: Why does ELBO decrease?')
print('='*80)

# Simple setup with 3 outputs
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

print('\nTracking ELBO components through epochs:')
print('Epoch | ExpLL      | BondKL   | NodeKL    | TauKL  | ELBO       | ΔELBO      | Status')
print('-'*100)

elbo_prev = None
for epoch in range(8):
    # Compute BEFORE update
    exp_ll = model.compute_expected_log_likelihood()
    bond_kl = model.compute_bond_kl(verbose=False)
    node_kl = model.compute_node_kl(verbose=False)
    tau_kl = torch.distributions.kl_divergence(model.q_tau.forward(), model.p_tau.forward()).item()
    elbo = exp_ll - (bond_kl + node_kl + tau_kl)
    
    delta_str = '---' if elbo_prev is None else f'{elbo - elbo_prev:+11.2f}'
    status = '' if elbo_prev is None else ('✓' if elbo >= elbo_prev else '✗ BUG!')
    
    print(f'{epoch:5d} | {exp_ll:10.2f} | {bond_kl:8.2f} | {node_kl:9.2f} | {tau_kl:6.2f} | {elbo:10.2f} | {delta_str:>11} | {status}')
    
    # Check if NodeKL exploded
    if node_kl > 1000:
        print(f'\n  ⚠️  NodeKL exploded to {node_kl:.2f}! Investigating...')
        print(f'  Node KLs individually:')
        for node_tag in ['Node1', 'Node2', 'Node3']:
            # Manually compute KL for this node
            try:
                kl = model.compute_node_kl(verbose=True)
                break
            except:
                pass
    
    elbo_prev = elbo
    
    # Do updates
    for node_tag in ['Node1', 'Node2', 'Node3']:
        model.update_sigma_node(node_tag)
        model.update_mu_node(node_tag)
    
    bonds = [i for i in model.mu.ind_map if i != model.batch_dim]
    for bond_tag in bonds:
        model.update_bond(bond_tag)
    
    model.update_tau()

print('\n' + '='*80)
print('CONCLUSION:')
print('If ELBO ever decreases (✗), there is a bug in the ELBO formula or updates!')
print('='*80)
