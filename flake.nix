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
        pygraphviz
        pandas
        torchvision
        torch
        jedi-language-server
        scipy
        matplotlib
        scikit-learn
        pytest
        jax
        quimb
      ]);

    opencodecfg = pkgs.writeText "opencodecfg.jsonc" ''
      {
        "$schema": "https://opencode.ai/config.json",
        "tui": {
          "scroll_speed": 3,
          "scroll_acceleration": {
            "enabled": true
          }
        },
        "mcp": {
          "serena": {
            "type": "local",
            "command": [
              "${pkgs.uv}/bin/uvx",
              "--python",
              "${python}/bin/python",
              "--from",
              "git+https://github.com/oraios/serena",
              "serena",
              "start-mcp-server",
              "--context",
              "ide-assistant"
            ],
            "enabled": true
          },
          "docs-mcp": {
            "type": "remote",
            "url": "http://127.0.0.1:6280/sse",
            "enabled": true
          }
        }
      }
    '';
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ];
      packages = [
        pythonWithNixPkgs
        pkgs.uv
        pkgs.opencode
      ];

      shellHook = ''
        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib]}"
        export OPENCODE_CONFIG=${opencodecfg}
        export PROJECT_DIR=$PWD
        # --- PODMAN SERVICE ---
        DOCS_CONTAINER_NAME="docs-mcp-server-dev"
        DATA_DIR="$PWD/.docs-data"
        mkdir -p "$DATA_DIR"
        # Load GEMINI_API_KEY from sops-nix secrets
        GEMINI_API_KEY_FILE="$HOME/.config/sops-nix/secrets/geminiApi"
        if [ -f "$GEMINI_API_KEY_FILE" ]; then
          export GEMINI_API_KEY=$(cat "$GEMINI_API_KEY_FILE")
          echo "✓ Loaded GEMINI_API_KEY from sops-nix secrets"
        else
          echo "⚠️  Warning: GEMINI_API_KEY file not found at $GEMINI_API_KEY_FILE"
          echo "   Zen MCP server may not work properly."
        fi

        cleanup() {
          echo "Stopping docs-mcp-server..."
          ${pkgs.podman}/bin/podman stop "$DOCS_CONTAINER_NAME" >/dev/null 2>&1 || true
        }
        trap cleanup EXIT

        Start server if not running
        if ! podman ps --format "{{.Names}}" | grep -q "^$DOCS_CONTAINER_NAME$"; then
          echo "🚀 Starting docs-mcp-server..."
          podman run --rm -d \
            --name "$DOCS_CONTAINER_NAME" \
            -v "$DATA_DIR:/data" \
            -p 6280:6280 \
            ghcr.io/arabold/docs-mcp-server:latest \
            --protocol http --host 0.0.0.0 --port 6280 --no-telemetry
          sleep 2
        else
          echo "✓ docs-mcp-server is running."
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
        if ! uv pip list | grep -q "^aim "; then
          echo "Installing aim via UV..."
          uv pip install aim
        fi

        # uv sync
        echo "✓ Environment ready!"
        echo "  Python: $(which python)"
        zsh
      '';
    };
  };
}
