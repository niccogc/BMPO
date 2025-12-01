# Bayesian MPO Experiments

## Standardized Experiment Framework

All experiment scripts follow a standardized format using environment variables for hyperparameters.
**No need to edit files** - just set environment variables!

## Quick Start

### 1D Polynomial Regression
```bash
# Basic run (3 blocks, 40 iterations)
python visualize_predictions_standardized.py

# Customize hyperparameters
NUM_BLOCKS=5 BOND_DIM=8 MAX_ITER=100 python visualize_predictions_standardized.py

# With automatic trimming
TRIM_THRESHOLD=10.0 NUM_BLOCKS=4 python visualize_predictions_standardized.py
```

### 2D Surface Regression
```bash
# Basic 2D surface
PROBLEM=2d python visualize_predictions_standardized.py

# High-capacity model
PROBLEM=2d NUM_BLOCKS=5 BOND_DIM=8 MAX_ITER=100 python visualize_predictions_standardized.py
```

## Environment Variables Reference

### Model Architecture
| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_BLOCKS` | 3 | Number of MPO blocks |
| `BOND_DIM` | 6 | Bond dimension (rank) |

### Training
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ITER` | 40 | Training iterations |
| `TRIM_THRESHOLD` | 0 | Trimming threshold (0 = no trimming) |

### Data Generation
| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_SAMPLES` | 100 | Number of training samples |
| `NOISE_STD` | 0.1 | Noise standard deviation |
| `PROBLEM` | '1d' | Problem type: '1d' or '2d' |

### Reproducibility
| Variable | Default | Description |
|----------|---------|-------------|
| `SEED` | 42 | Random seed for data/model |
| `PRIOR_SEED` | 42 | Random seed for priors |

### Output
| Variable | Default | Description |
|----------|---------|-------------|
| `SAVE_PLOTS` | 1 | Save plots (1=yes, 0=no) |
| `OUTPUT_PREFIX` | 'predictions' | Output filename prefix |

## Example Workflows

### Parameter Sweep
```bash
# Test different model sizes
for blocks in 3 4 5; do
    NUM_BLOCKS=$blocks python visualize_predictions_standardized.py
done

# Test different bond dimensions
for dim in 4 6 8 10; do
    BOND_DIM=$dim OUTPUT_PREFIX="sweep_d${dim}" python visualize_predictions_standardized.py
done
```

### Reproducibility
```bash
# Same seed = same results
SEED=123 PRIOR_SEED=456 python visualize_predictions_standardized.py

# Re-run with same settings
SEED=123 PRIOR_SEED=456 python visualize_predictions_standardized.py
```

### Trimming Experiments
```bash
# Compare with/without trimming
TRIM_THRESHOLD=0 OUTPUT_PREFIX="no_trim" python visualize_predictions_standardized.py
TRIM_THRESHOLD=10 OUTPUT_PREFIX="with_trim" python visualize_predictions_standardized.py
```

## Output Files

Plots are saved with descriptive names:
- Format: `{OUTPUT_PREFIX}_{PROBLEM}_b{NUM_BLOCKS}_d{BOND_DIM}.png`
- Examples:
  - `predictions_1d_b3_d6.png` - Default 1D polynomial
  - `predictions_2d_b5_d8.png` - 2D surface with 5 blocks, dim 8
  - `sweep_d10_1d_b4_d10.png` - Custom prefix from sweep

## Available Scripts

### `visualize_predictions_standardized.py`
Main visualization script for 1D and 2D regression with uncertainty quantification.

### `debug_blocks.py` 
Debug script to check block updates (also uses env vars).

More scripts to be added following the same standardized format.

## Key Fix: Diagonal Σ Initialization

**Important**: The Σ-blocks are now initialized as diagonal matrices scaled by `1/num_blocks`.
This fixes the collapse problem that occurred with 4+ blocks in previous versions.

Before: Σ-blocks were initialized as μ ⊗ μ^T causing exponential growth.
After: Σ-blocks are initialized as `(0.1/num_blocks) * I` (diagonal, scaled).

This allows models with 4, 5, or more blocks to train successfully!
