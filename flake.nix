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

    opencodecfg = pkgs.writeText "opencodecfg.jsonc" ''
      {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
          "serena": {
            "type": "local",
            "command": [
              "uvx",
              "--from",
              "git+https://github.com/oraios/serena",
              "serena",
              "start-mcp-server",
              "--context",
              "ide-assistant",
              "--project",
              "/etc/nixos/ai"
            ],
            "enabled": true
          },
          "nixos": {
            "type": "local",
            "command": [
              "mcp-nixos"
            ],
            "enabled": true
          },
          "zen": {
            "type": "local",
            "command": [
              "${pkgs.uv}/bin/uvx",
              "--from",
              "git+https://github.com/BeehiveInnovations/zen-mcp-server.git",
              "zen-mcp-server"
            ],
            "enabled": true
          }
        }
      }
    '';

    geminicfg = pkgs.writeText "gemini-settings.json" ''
      {
        "mcpServers": {
          "serena": {
            "type": "local",
            "command": [
              "uvx",
              "--from",
              "git+https://github.com/oraios/serena",
              "serena",
              "start-mcp-server",
              "--context",
              "ide-assistant",
              "--project",
              "/etc/nixos/ai"
            ],
            "enabled": true
          },
          "nixos": {
            "type": "local",
            "command": [
              "mcp-nixos"
            ],
            "enabled": true
          },
        }
      }
    '';
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pythonWithNixPkgs
        pkgs.uv
        pkgs.gemini-cli-bin
        pkgs.opencode
      ];

      shellHook = ''
        export OPENCODE_CONFIG=${opencodecfg}
        export PROJECT_DIR=$PWD
        export GEMINI_SETTINGS=${geminicfg}

        # Load GEMINI_API_KEY from sops-nix secrets
        GEMINI_API_KEY_FILE="$HOME/.config/sops-nix/secrets/geminiApi"
        if [ -f "$GEMINI_API_KEY_FILE" ]; then
          export GEMINI_API_KEY=$(cat "$GEMINI_API_KEY_FILE")
          echo "✓ Loaded GEMINI_API_KEY from sops-nix secrets"
        else
          echo "⚠️  Warning: GEMINI_API_KEY file not found at $GEMINI_API_KEY_FILE"
          echo "   Zen MCP server may not work properly."
        fi
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
        # if ! uv pip list | grep -q "^aim "; then
        #   echo "Installing aim via UV..."
        #   uv pip install aim
        # fi

        # uv sync
        echo "✓ Environment ready!"
        echo "  Python: $(which python)"
        zsh
      '';
    };
  };
}
