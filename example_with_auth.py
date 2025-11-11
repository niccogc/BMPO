"""Example: Using AIM with authentication for remote tracking"""

import os
import torch
import jax.numpy as jnp

# IMPORTANT: Import Run from aim_auth instead of aim
from aim_auth import Run

# Set your authentication token (you'd normally set this in your environment or secrets)
os.environ['AIM_AUTH_TOKEN'] = 'your-secret-token-here'

# Alternative: Use Cloudflare Access tokens
# os.environ['CF_ACCESS_CLIENT_ID'] = 'your-client-id'
# os.environ['CF_ACCESS_CLIENT_SECRET'] = 'your-client-secret'

# Alternative: Use custom headers via JSON
# os.environ['AIM_CUSTOM_HEADERS'] = '{"X-API-Key": "your-key", "X-Custom": "value"}'

print("=" * 60)
print("AIM Remote Tracking with Authentication Example")
print("=" * 60)

# For local testing without remote server, use local repo
# run = Run()

# For remote server with authentication (replace with your server URL)
# run = Run(repo='aim://your-server.cloudflare.com:53800')

# For this demo, we'll use local tracking
run = Run()

print(f"\nRun hash: {run.hash}")
print("\nTracking metrics with authentication enabled...")

# Simulate training with PyTorch
torch_losses = torch.randn(5).abs()
for i, loss in enumerate(torch_losses):
    run.track(float(loss), name="torch_loss", step=i)
    print(f"  Step {i}: torch_loss = {loss:.4f}")

# Simulate training with JAX
jax_losses = jnp.array([0.5, 0.4, 0.3, 0.2, 0.1])
for i, loss in enumerate(jax_losses):
    run.track(float(loss), name="jax_loss", step=i)
    print(f"  Step {i}: jax_loss = {loss:.4f}")

print("\n" + "=" * 60)
print("Done! All metrics tracked successfully.")
print("View results with: aim up")
print("=" * 60)
