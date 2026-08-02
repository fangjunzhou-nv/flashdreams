{
  description = "FlashDreams development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    with flake-utils.lib;
    eachSystem [
      system.x86_64-linux
      system.aarch64-darwin
    ]
      (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              python312
              uv
            ];

            shellHook = ''
              # Always use the Nix-provided interpreter. Generic uv-managed
              # Python binaries do not run on NixOS without nix-ld.
              export UV_PYTHON_DOWNLOADS=never

              if [ -d .venv ]; then
                source .venv/bin/activate
                export PATH="$PWD/.venv/bin:$PATH"
              else
                echo "Environment not initialized. Run: uv sync --extra dev --extra runners"
              fi
            '';
          };
        });
}
