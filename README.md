# Bayesian MPO with Hybrid Nix + UV Setup

This project uses a hybrid Python environment:
- **Nix** manages: `torch`, `jax`, and `quimb` (heavy scientific packages)
- **UV** manages: `aim` (which is difficult to package with Nix)

## How It Works

The setup uses UV to create a virtual environment and PYTHONPATH to access Nix packages:
1. Nix builds `torch`, `jax`, and `quimb` with all optimizations
2. UV creates a lightweight venv and installs `aim`
3. PYTHONPATH is set to include Nix packages
4. Both sets of packages are available in the same Python environment

## Usage

### Enter the development environment:
```bash
nix develop
```

This will automatically:
- Create a UV virtual environment (if needed)
- Install `aim` via UV (if needed)
- Activate the environment with both Nix and UV packages available

### Run your code:
```bash
# Already in the activated environment after 'nix develop'
python main.py
```

### Install additional UV packages:
```bash
uv pip install <package>
```

### Update Nix packages:
Edit the `myPkgs` list in `flake.nix`:
```nix
myPkgs = p: with p; [torch quimb jax your-new-package];
```

Then run:
```bash
nix flake update
```

## File Structure

- `flake.nix` - Defines Nix packages and development shell
- `pyproject.toml` - Defines UV-managed packages (aim)
- `.venv/` - UV virtual environment (gitignored)
- `uv.lock` - UV lockfile for reproducibility

## Benefits

✓ Heavy packages (torch, jax, quimb) built by Nix with optimizations  
✓ Problematic packages (aim) handled by UV  
✓ Single Python environment with all packages  
✓ Reproducible builds via Nix flakes  
✓ Fast dependency resolution via UV  

## Technical Details

**The key to making Nix + UV work together:**

1. **Single Python interpreter**: Only include `pythonWithPackages` in the shell (not bare `python`), avoiding PATH conflicts
2. **PYTHONPATH for Nix packages**: Nix packages are accessible via `PYTHONPATH` in the UV venv
3. **LD_LIBRARY_PATH for native deps**: `aim`'s native dependency `aimrocks` needs `libz.so.1`, provided by adding `zlib` and stdenv to the shell
4. **UV venv for pip packages**: UV manages packages that are hard to build with Nix (like `aim`)

## Troubleshooting

**Q: Packages not found after adding to flake.nix?**  
A: Rebuild the environment: `exit` and `nix develop` again

**Q: Want to reset the UV environment?**  
A: `rm -rf .venv` then `nix develop`

**Q: Check which packages come from where?**  
A:
```bash
# Nix packages location
echo $NIX_PYTHON_SITE_PACKAGES

# UV packages location  
uv pip list
```
