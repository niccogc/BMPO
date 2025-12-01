import torch
import numpy as np
import os
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train
import aim
from aim_auth import Run

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
NUM_BLOCKS_RANGE = [2, 3, 4]
BOND_DIM_RANGE = [2,3,4]
MAX_ITER = 5
SEED = 42
PRIOR_SEED = 42
TEST_SIZE = 0.2

# AIM Tracking setup
AIM_REPO = 'aim://192.168.5.5:5800'

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
print("Loading Diabetes dataset...")
diabetes = fetch_california_housing()
X, y_raw = diabetes.data, diabetes.target

# Add a bias term (intercept) to the features
X_b = np.hstack([np.ones((X.shape[0], 1)), X])
y_raw = y_raw.reshape(-1, 1)

# Split data
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X_b, y_raw, test_size=TEST_SIZE, random_state=SEED
)

# Standardize features (excluding the bias term)
scaler_X = StandardScaler()
X_train[:, 1:] = scaler_X.fit_transform(X_train[:, 1:])
X_test[:, 1:] = scaler_X.transform(X_test[:, 1:])

# Standardize target
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train_raw)
y_test = scaler_y.transform(y_test_raw)

# Convert to torch tensors
X_train = torch.from_numpy(X_train).double()
y_train = torch.from_numpy(y_train).double().squeeze()
X_test = torch.from_numpy(X_test).double()
y_test = torch.from_numpy(y_test).double().squeeze()

INPUT_DIM = X_train.shape[1]
print(f"Data loaded and preprocessed. Input dimensions: {INPUT_DIM}")
print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape:  X={X_test.shape}, y={y_test.shape}")
print("-" * 70)

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================
print("Starting regression testing experiment...")

for num_blocks in NUM_BLOCKS_RANGE:
    for bond_dim in BOND_DIM_RANGE:
        print(f"Running experiment for: Blocks={num_blocks}, BondDim={bond_dim}")

        run = Run(repo=AIM_REPO, experiment='regression_testing')
        
        run['hparams'] = {
            'num_blocks': num_blocks,
            'bond_dim': bond_dim,
            'max_iter': MAX_ITER,
            'dataset': 'california_housing',
            'seed': SEED,
            'prior_seed': PRIOR_SEED,
        }

        bmpo = create_bayesian_tensor_train(
            num_blocks=num_blocks,
            bond_dim=bond_dim,
            input_features=INPUT_DIM,
            output_shape=1,
            dtype=torch.float64,
            seed=SEED,
            random_priors=True,
            prior_seed=PRIOR_SEED
        )

        for iteration in range(MAX_ITER):
            # --- Training Step ---
            for block_idx in range(num_blocks):
                bmpo.update_block_variational(block_idx, X_train, y_train)
            for label in bmpo.mu_mpo.distributions.keys():
                bmpo.update_bond_variational(label)
            bmpo.update_tau_variational(X_train, y_train)

            # --- Metrics Tracking (at every iteration) ---
            context = {'subset': 'train'}
            mu_train = torch.as_tensor(bmpo.forward_mu(X_train, to_tensor=True)).squeeze()
            mse_train = ((mu_train - y_train)**2).mean().item()
            r2_train = 1 - mse_train / y_train.var().item()
            run.track(mse_train, name='mse', step=iteration, context=context)
            run.track(r2_train, name='r2', step=iteration, context=context)

            context = {'subset': 'test'}
            mu_test = torch.as_tensor(bmpo.forward_mu(X_test, to_tensor=True)).squeeze()
            sigma_test = torch.as_tensor(bmpo.forward_sigma(X_test, to_tensor=True)).squeeze()
            
            mse_test = ((mu_test - y_test)**2).mean().item()
            rmse_test = np.sqrt(mse_test)
            r2_test = 1 - mse_test / y_test.var().item()
            
            tau_mean = bmpo.get_tau_mean().item()
            aleatoric_var = 1.0 / tau_mean
            total_var = sigma_test + aleatoric_var
            
            run.track(mse_test, name='mse', step=iteration, context=context)
            run.track(rmse_test, name='rmse', step=iteration, context=context)
            run.track(r2_test, name='r2', step=iteration, context=context)
            run.track(tau_mean, name='tau_mean', step=iteration, context=context)
            run.track(sigma_test.mean().item(), name='mean_epistemic_variance', step=iteration, context=context)
            run.track(total_var.mean().item(), name='mean_total_variance', step=iteration, context=context)

            if (iteration + 1) % 5 == 0:
                print(f"  Iter {iteration+1:3d}: Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | E[τ]={tau_mean:.2f}")
        
        print("-" * 70)
        run.close()

print("Experiment finished.")
