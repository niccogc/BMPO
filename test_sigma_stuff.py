import torch
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
torch.set_default_dtype(torch.float64)
N = 100
x_raw = torch.rand(N, 1)
y_raw = torch.cat([x_raw**2, x_raw**3], dim=1)
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)
loader = Inputs(inputs=[x_features], outputs=[y_raw], outputs_labels=['y'], input_labels=['x1', 'x2', 'x3'], batch_dim='s', batch_size=50)
t1 = qt.Tensor(data=torch.randn(2, 4, 4)/10, inds=('x1', 'b1', 'b3'), tags={'Node1'})
t2 = qt.Tensor(data=torch.randn(4, 2, 4)/10, inds=('b1', 'x2', 'b2'), tags={'Node2'})
t3 = qt.Tensor(data=torch.randn(4, 2, 4, 2)/10, inds=('b2', 'x3', 'b3', 'y'), tags={'Node3'})
mu_tn = qt.TensorNetwork([t1, t2, t3])
model = BTN(mu=mu_tn, data_stream=loader, batch_dim='s', method='cholesky')
# Actually run the update
print('Testing sigma update for Node3...')
try:
    model.update_sigma_node('Node3')
    print('✓ SUCCESS!')
    print(f'  Sigma indices: {model.sigma["Node3"].inds}')
    print(f'  Sigma shape: {model.sigma["Node3"].shape}')
except Exception as e:
    print(f'✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
