"""Demo showing Nix packages (torch, jax) working with UV package (aim)"""
import torch
import jax
import jax.numpy as jnp
import aim

print("✓ Successfully imported:")
print(f"  - torch {torch.__version__} (from Nix)")
print(f"  - jax {jax.__version__} (from Nix)")
print(f"  - aim (from UV)")

# Create some dummy training data
print("\n📊 Creating dummy training run with Aim...")
run = aim.Run()

# Simulate training with PyTorch tensor
torch_data = torch.randn(5)
for i, loss in enumerate(torch_data):
    run.track(float(loss), name="torch_loss", step=i)

# Simulate training with JAX
jax_data = jnp.array([0.5, 0.4, 0.3, 0.2, 0.1])
for i, loss in enumerate(jax_data):
    run.track(float(loss), name="jax_loss", step=i)

print("✓ Tracked values to Aim")
print(f"✓ Run hash: {run.hash}")
print("\nView results: aim up")
