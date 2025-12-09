# Bayesian MPO Experiments Guide

## Standardized Experiment Framework

**ALL experiment scripts MUST use environment variables for hyperparameters.**

See `EXPERIMENTS_README.md` for full documentation.

## Key Scripts

### `visualize_predictions_standardized.py`
Main visualization script with env var support.

Usage:
```bash
# 1D polynomial
python visualize_predictions_standardized.py

# 2D surface  
PROBLEM=2d python visualize_predictions_standardized.py

# Custom configuration
NUM_BLOCKS=5 BOND_DIM=8 MAX_ITER=100 TRIM_THRESHOLD=10 python visualize_predictions_standardized.py
```

## Standard Environment Variables

- `NUM_BLOCKS` (default: 3) - Number of MPO blocks
- `BOND_DIM` (default: 6) - Bond dimension  
- `MAX_ITER` (default: 40) - Training iterations
- `TRIM_THRESHOLD` (default: 0) - Trimming threshold (0=no trim)
- `NUM_SAMPLES` (default: 100) - Training samples
- `NOISE_STD` (default: 0.1) - Noise level
- `PROBLEM` (default: '1d') - Problem type: '1d', '2d'
- `SEED` (default: 42) - Random seed
- `PRIOR_SEED` (default: 42) - Prior seed
- `SAVE_PLOTS` (default: 1) - Save plots flag
- `OUTPUT_PREFIX` (default: 'predictions') - Output filename prefix

## Important Fix: Diagonal Σ Initialization

**Fixed in `tensor/bayesian_mpo.py`:**

The Σ-blocks are now initialized as diagonal matrices scaled by `1/num_blocks`:
```python
initial_variance = 0.1 / num_blocks
sigma_matrix = initial_variance * torch.eye(d)
```

This replaces the old initialization `Σ = μ ⊗ μ^T + σ²I` which caused exponential growth with multiple blocks.

**Result**: Models with 4, 5+ blocks now train successfully without collapse!

## Outputs

Plots saved as: `{OUTPUT_PREFIX}_{PROBLEM}_b{NUM_BLOCKS}_d{BOND_DIM}.png`

Examples:
- `predictions_1d_b3_d6.png` - 1D polynomial, 3 blocks, dim 6
- `predictions_2d_b5_d8.png` - 2D surface, 5 blocks, dim 8
