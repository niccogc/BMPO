{
  description = "Bayesian Env with Aim";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    quimb-flake = {
      url = "github:niccogc/quimbflake";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    quimb-flake,
  }: let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
      inherit system;
      overlays = [quimb-flake.overlays.default];
    };

    python = pkgs.python312;
    pythonWithNixPkgs = python.withPackages (ps:
      with ps; [
        torch
        jax
        quimb
      ]);
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pythonWithNixPkgs
        pkgs.uv
      ];
      
      shellHook = ''
        # Get the Nix Python site-packages path
        export NIX_PYTHON_SITE_PACKAGES="${pythonWithNixPkgs}/${pythonWithNixPkgs.sitePackages}"
        
        # Create UV venv if it doesn't exist
        if [ ! -d .venv ]; then
          echo "Creating UV virtual environment..."
          uv venv --python ${python}/bin/python
        fi
        
        # Activate the venv
        source .venv/bin/activate
        
        # Add Nix packages to PYTHONPATH so they're available in the venv
        export PYTHONPATH="$NIX_PYTHON_SITE_PACKAGES:$PYTHONPATH"
        
        # Install aim if not already installed
        if ! uv pip list | grep -q "^aim "; then
          echo "Installing aim via UV..."
          uv pip install aim
        fi
        
        echo "✓ Environment ready!"
        echo "  Nix packages: torch, jax, quimb"
        echo "  UV packages: aim"
        echo "  Python: $(which python)"
      '';
    };
  };
}
