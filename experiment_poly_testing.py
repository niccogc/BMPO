import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train
import aim # Added import aim
from aim_auth import Run # Using aim_auth as per AIM_RUN_INSTRUCTION.md
# TODO: the r2 and mse should be tracked for each iteration not only the final value.
# ============================================================================
# HELPER FUNCTIONS FOR POLYNOMIAL DATA GENERATION
# ============================================================================
def generate_polynomial_data(degree, num_samples, noise_std, seed):
    torch.manual_seed(seed)
    x = torch.rand(num_samples, dtype=torch.float64) * 2 - 1
    
    # Generate random coefficients for the polynomial
    # Ensure the highest degree coefficient is not zero for a true 'degree' polynomial
    coeffs = torch.randn(degree + 1, dtype=torch.float64)
    if degree > 0:
        while coeffs[degree] == 0: # Ensure highest degree coeff is not zero
            coeffs[degree] = torch.randn(1, dtype=torch.float64)
    
    y_true = torch.zeros_like(x)
    poly_str = ""
    for i in range(degree + 1):
        y_true += coeffs[i] * (x**i)
        if i == 0:
            poly_str += f"{coeffs[i]:.2f}"
        else:
            poly_str += f" + {coeffs[i]:.2f}x^{i}"
    
    y = y_true + noise_std * torch.randn(num_samples, dtype=torch.float64)
    X = torch.stack([x**i for i in range(degree + 1)], dim=1) # Features up to degree

    # Test grid
    x_test = torch.linspace(-1.2, 1.2, 200, dtype=torch.float64)
    y_test_true = torch.zeros_like(x_test)
    for i in range(degree + 1):
        y_test_true += coeffs[i] * (x_test**i)
    X_test = torch.stack([x_test**i for i in range(degree + 1)], dim=1)

    return X, y, x_test, y_test_true, X_test, poly_str, coeffs

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
NUM_BLOCKS_RANGE = [2, 3, 4, 5]
BOND_DIM_RANGE = [4, 5, 6, 7]
POLYNOMIAL_DEGREE_RANGE = [2, 3, 4, 5]
MAX_ITER = 10 # Changed to 10 as per user request
NUM_SAMPLES = 100
NOISE_STD = 0.1
SEED = 42
PRIOR_SEED = 42
SAVE_PLOTS = 0 # Set to 0 for automated testing, 1 to save plots

# AIM Tracking setup
AIM_REPO = 'aim://192.168.5.5:5800' # As per AIM_RUN_INSTRUCTION.md

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================
print("Starting polynomial testing experiment...")

for num_blocks in NUM_BLOCKS_RANGE:
    for bond_dim in BOND_DIM_RANGE:
        for poly_degree in POLYNOMIAL_DEGREE_RANGE:
            print(f"Running experiment for: Blocks={num_blocks}, BondDim={bond_dim}, Degree={poly_degree}")

            # Initialize AIM Run
            run = Run(repo=AIM_REPO, experiment='poly_testing')
            
            # Log hyperparameters
            run['hparams'] = {
                'num_blocks': num_blocks,
                'bond_dim': bond_dim,
                'poly_degree': poly_degree,
                'max_iter': MAX_ITER,
                'num_samples': NUM_SAMPLES,
                'noise_std': NOISE_STD,
                'seed': SEED,
                'prior_seed': PRIOR_SEED,
            }

            # Generate data for the current polynomial degree
            X, y, x_test, y_test_true, X_test, poly_str, coeffs = generate_polynomial_data(
                poly_degree, NUM_SAMPLES, NOISE_STD, SEED
            )
            INPUT_DIM = X.shape[1] # Update INPUT_DIM based on polynomial degree
            run['polynomial_string'] = poly_str
            run['polynomial_coefficients'] = coeffs.tolist()

            # Create and train model
            bmpo = create_bayesian_tensor_train(
                num_blocks=num_blocks,
                bond_dim=bond_dim,
                input_features=INPUT_DIM,
                output_shape=1,
                constrict_bond=False,
                tau_alpha=torch.tensor(2.0, dtype=torch.float64),
                tau_beta=torch.tensor(1.0, dtype=torch.float64),
                dtype=torch.float64,
                seed=SEED,
                random_priors=True,
                prior_seed=PRIOR_SEED
            )

            for iteration in range(MAX_ITER):
                for block_idx in range(num_blocks):
                    bmpo.update_block_variational(block_idx, X, y)
                
                for label in bmpo.mu_mpo.distributions.keys():
                    bmpo.update_bond_variational(label)
                
                bmpo.update_tau_variational(X, y)
                
                if (iteration + 1) % 10 == 0:
                    mu_pred_train = torch.as_tensor(bmpo.forward_mu(X, to_tensor=True)) # Explicitly convert to tensor
                    mse_train = ((mu_pred_train.squeeze() - y)**2).mean().item()
                    tau = bmpo.get_tau_mean().item()
                    run.track(mse_train, name='mse_train', step=iteration)
                    run.track(tau, name='tau_mean', step=iteration)
                    print(f"  Iter {iteration+1:3d}: MSE={mse_train:.6f}, E[τ]={tau:.2f}")

            # Evaluate
            mu_pred_test = torch.as_tensor(bmpo.forward_mu(X_test, to_tensor=True)) # Explicitly convert to tensor
            sigma_pred_test = torch.as_tensor(bmpo.forward_sigma(X_test, to_tensor=True)) # Explicitly convert to tensor
            
            mu_pred_test = mu_pred_test.squeeze().detach().cpu()
            sigma_pred_test = sigma_pred_test.squeeze().detach().cpu()
            
            tau_mean = bmpo.get_tau_mean().item()
            aleatoric_var = 1.0 / tau_mean
            total_var = sigma_pred_test + aleatoric_var
            total_std = torch.sqrt(total_var).detach().cpu()
            epistemic_std = torch.sqrt(sigma_pred_test).detach().cpu()
            
            residuals = mu_pred_test - y_test_true.detach().cpu()
            rmse = torch.sqrt((residuals**2).mean()).item()
            ss_res = (residuals**2).sum().item()
            ss_tot = ((y_test_true.detach().cpu() - y_test_true.detach().cpu().mean())**2).sum().item()
            r2 = 1 - (ss_res / ss_tot)

            # Log evaluation metrics
            run.track(rmse, name='rmse_test')
            run.track(r2, name='r2_test')
            run.track(tau_mean, name='final_tau_mean')
            run.track(epistemic_std.mean().item(), name='mean_epistemic_std')
            run.track(total_std.mean().item(), name='mean_total_std')

            print(f"  Test RMSE: {rmse:.5f}, Test R²: {r2:.5f}")
            print("-" * 50)

            # Plotting (optional)
            if SAVE_PLOTS:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                ax = axes[0]
                ax.scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), alpha=0.5, s=20, c='gray', label='Training data')
                ax.plot(x_test.detach().cpu().numpy(), y_test_true.detach().cpu().numpy(), 'k-', linewidth=2, label='True')
                ax.plot(x_test.detach().cpu().numpy(), mu_pred_test.detach().cpu().numpy(), 'r-', linewidth=2, label='Predicted mean')
                ax.fill_between(x_test.detach().cpu().numpy(),
                                (mu_pred_test - 2*total_std).detach().cpu().numpy(),
                                (mu_pred_test + 2*total_std).detach().cpu().numpy(),
                                alpha=0.3, color='red', label='±2σ total')
                ax.fill_between(x_test.detach().cpu().numpy(),
                                (mu_pred_test - 2*epistemic_std).detach().cpu().numpy(),
                                (mu_pred_test + 2*epistemic_std).detach().cpu().numpy(),
                                alpha=0.5, color='blue', label='±2σ epistemic')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f'Predictions (R²={r2:.3f})', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                ax = axes[1]
                ax.scatter(x_test.detach().cpu().numpy(), residuals.detach().cpu().numpy(), alpha=0.5, s=10, c='purple')
                ax.axhline(y=0, color='k', linestyle='--')
                ax.fill_between(x_test.detach().cpu().numpy(),
                                (-2*total_std).detach().cpu().numpy(),
                                (2*total_std).detach().cpu().numpy(),
                                alpha=0.2, color='red')
                ax.set_xlabel('x')
                ax.set_ylabel('Residual')
                ax.set_title(f'Residuals (RMSE={rmse:.4f})', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = f'poly_testing_b{num_blocks}_d{bond_dim}_deg{poly_degree}.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                run.track(aim.Image(filename), name='prediction_plot')
                os.remove(filename) # Clean up the plot file
                plt.close(fig) # Close the figure to free memory

print("Experiment finished.")
